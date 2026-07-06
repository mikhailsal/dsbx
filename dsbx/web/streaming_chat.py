"""SSE glue for the chat-simulation generate path.

Chat-only providers (NIM / OpenRouter) can't take a raw rendered prompt,
so the chat-mode UI sends structured ``messages[]`` and this module
drives :meth:`OpenAICompatBackend.stream_chat_native` into the exact
same ``step* -> usage -> done`` SSE frame sequence that
:func:`dsbx.web.streaming.stream_generate` produces -- the browser's
decode table renders both without knowing which path ran.

Kept out of ``web/streaming.py`` (grandfathered size ceiling) and out of
``web/app.py`` (same); the route in ``app.py`` calls
:func:`chat_stream_response` and stays two lines long.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from dsbx.core import usage as usage_mod
from dsbx.core.backend import Backend
from dsbx.server.schemas import genstep_to_wire
from dsbx.web.backends import BackendRegistry
from dsbx.web.deps import use_backend
from dsbx.web.schemas_chat import ChatGenerateRequest
from dsbx.web.streaming import sse_frame

log = logging.getLogger("dsbx.web.streaming_chat")


def chat_stream_response(
    registry: BackendRegistry,
    req: ChatGenerateRequest,
    sampler_name: str,
    sampler_params: dict,
) -> StreamingResponse:
    """Validate a ``messages[]`` generate request and open the SSE stream.

    Guard matrix (the other two combinations -- no messages -- stay on
    the historical prompt path in ``app.py``):

    - chat-capable backend + ``messages`` -> stream via /chat/completions;
    - non-chat backend + ``messages`` -> 400 telling the client to render
      the template locally and send ``prompt`` (one source of truth).
    """
    with use_backend(registry, req.backend, model=req.model) as backend:
        caps = backend.capabilities
        if not bool(getattr(caps, "supports_chat_stream", False)):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"backend {req.backend!r} does not stream chat messages "
                    "natively; render the chat template client-side and send "
                    "the result as `prompt` instead of `messages`."
                ),
            )
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages must be a non-empty list.")
        if not backend.supports_chat_sampler(sampler_name):  # type: ignore[attr-defined]
            allowed = ", ".join(getattr(caps, "chat_samplers", ())) or "greedy"
            raise HTTPException(
                status_code=400,
                detail=(
                    f"sampler {sampler_name!r} cannot run server-side on a "
                    "chat-only provider (no /chat/completions analogue); "
                    f"use one of: {allowed} in simulation mode."
                ),
            )

    def _body() -> Iterator[bytes]:
        # Re-acquiring the backend can fail even though the validation
        # pass above succeeded (a concurrent model unload / registry
        # change between the two lock windows). At this point the 200
        # header is already on the wire, so raising would just drop the
        # connection mid-SSE -- surface the failure through the same
        # terminating ``done`` frame contract errors inside the stream
        # use.
        try:
            with use_backend(registry, req.backend, model=req.model) as backend:
                yield from _stream_chat_frames(backend, req, sampler_name, sampler_params)
        except HTTPException as exc:
            log.warning("dsbx-web: chat stream lost its backend: %s", exc.detail)
            yield sse_frame({"event": "done", "stop_reason": None, "error": str(exc.detail)})

    return StreamingResponse(
        _body(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _stream_chat_frames(
    backend: Backend,
    req: ChatGenerateRequest,
    sampler_name: str,
    sampler_params: dict,
) -> Iterator[bytes]:
    """Drive ``stream_chat_native`` into ``step* -> usage -> done`` frames.

    Mirrors the finalizer contract of ``stream_generate``: the usage sink
    is bound for the duration of the run (and always released), errors
    land in the terminating ``done`` frame instead of aborting the SSE
    body mid-stream, and ``completion_tokens`` falls back to the emitted
    step count when the provider didn't report usage.
    """
    usage: usage_mod.UsageSink = usage_mod.make_sink()
    completion_steps = 0
    error: str | None = None
    last_reason: str | None = None
    bound_usage = False
    if isinstance(backend, usage_mod.UsageAware):
        backend.set_active_usage(usage)
        bound_usage = True
    try:
        # Watch ids resolve through the local HF tokenizer when mapped;
        # unmapped chat-only backends intern the text (rank -1 everywhere,
        # rendered dim), which is the honest degradation.
        watch_ids = _resolve_chat_watches(backend, req)
        for gs in backend.stream_chat_native(  # type: ignore[attr-defined]
            [m.model_dump(exclude_none=True) for m in req.messages or []],
            sampler_name=sampler_name,
            sampler_params=sampler_params,
            max_tokens=int(req.max_tokens),
            top_k=int(req.top_k),
            seed=int(req.seed),
            tools=req.tools,
            stop_texts=list(req.stop_texts or []),
            watch_ids=watch_ids,
        ):
            yield sse_frame({"event": "step", "step": genstep_to_wire(gs).model_dump()})
            completion_steps += 1
            last_reason = gs.stop_reason
    except Exception as exc:
        log.exception("dsbx-web: chat stream errored")
        error = str(exc)
    finally:
        if bound_usage and isinstance(backend, usage_mod.UsageAware):
            backend.set_active_usage(None)

    if usage.get("completion_tokens") is None:
        usage["completion_tokens"] = int(completion_steps)
    usage.pop("perf_metrics", None)
    usage.pop("raw_output", None)
    yield sse_frame({"event": "usage", **usage})
    if error is not None:
        yield sse_frame({"event": "done", "stop_reason": last_reason, "error": error})
        return
    yield sse_frame({"event": "done", "stop_reason": last_reason})


def _resolve_chat_watches(backend: Backend, req: ChatGenerateRequest) -> list[int]:
    """Resolve watch texts/ids for the chat path (best-effort, never raises).

    ``watch_eos`` maps through the capabilities envelope like the prompt
    path; text watches take the first token id from the local tokenizer.
    Failures degrade to skipping the watch rather than killing the stream.
    """
    out: list[int] = []
    seen: set[int] = set()

    def _add(tid: int) -> None:
        if tid not in seen:
            seen.add(tid)
            out.append(tid)

    for txt in req.watch_texts or []:
        try:
            ids = backend.tokenize(txt)
        except Exception as exc:
            log.debug("chat watch text %r did not tokenize: %s", txt, exc)
            continue
        if ids:
            _add(int(ids[0]))
    for raw in req.watch_ids or []:
        try:
            _add(int(raw))
        except (TypeError, ValueError):
            continue
    if req.watch_eos:
        for tid in backend.capabilities.eos_token_ids:
            _add(int(tid))
    return out
