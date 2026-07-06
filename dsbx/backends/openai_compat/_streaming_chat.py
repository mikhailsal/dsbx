"""Chat-completions streaming mixin: the /chat/completions SSE decode loop.

The "simulation mode" engine for chat-only providers (NIM, OpenRouter).
Where template-capable backends receive the fully rendered chat template
as a raw ``prompt`` string, chat-only providers refuse raw prompts (or
silently rewrite them into messages server-side -- verified live for
OpenRouter), so the honest path is to send the structured ``messages[]``
the provider expects and stream back per-token ``top_logprobs``. Each
streamed token becomes a :class:`GenStep` shaped exactly like the
``/completions`` path produces, so the whole decode-table UI is reused
untouched.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

from dsbx.core import usage as usage_mod
from dsbx.core.engine import GenStep
from dsbx.core.samplers import CHAT_NATIVE_SAMPLERS

if TYPE_CHECKING:
    import httpx
    from tokenizers import Tokenizer

    from dsbx.core.config import ProviderConfig
    from dsbx.core.types import TokenCandidate

log = logging.getLogger(__name__)

# Emit-record shape shared with ``_genstep_from_emit_record``:
# ``(token_id_or_None, text, logprob, top_payload, sampling_mask_count)``.
_EmitRecord = tuple[int | None, str, float, Any, int | None]


class _ChatRecordAligner:
    """Reconcile the ``delta.content`` text stream with ``logprobs.content``.

    Well-behaved providers keep the two in lockstep (each chunk's delta
    text equals its logprob entries' concatenated tokens) and this class
    is a pass-through. Live probing found providers that do NOT (an
    OpenRouter/vLLM route streamed logprob entries offset from the text
    by several tokens, with the first tokens never getting entries at
    all). The contract this class restores: every character the provider
    put in ``delta.content`` is emitted exactly once and in order --
    backed by its logprob entry when one matches, as a NaN-logprob
    filler record otherwise. Entries whose token text never shows up in
    the text stream (typically the final EOT) are emitted at the end.
    """

    def __init__(self, make_filler: Any) -> None:
        self._make_filler = make_filler
        self._text = ""
        self._entries: list[tuple[str, _EmitRecord]] = []

    def push(self, delta_text: str, entries: list[tuple[str, _EmitRecord]]) -> list[_EmitRecord]:
        self._text += delta_text
        self._entries.extend(entries)
        return self._drain(final=False)

    def flush(self) -> list[_EmitRecord]:
        return self._drain(final=True)

    def _drain(self, *, final: bool) -> list[_EmitRecord]:
        out: list[_EmitRecord] = []
        while self._entries:
            token, record = self._entries[0]
            idx = self._text.find(token) if token else 0
            if idx == -1:
                if not final:
                    # The entry's text hasn't arrived in the delta stream
                    # yet -- wait for more chunks before deciding.
                    return out
                # Final: the token never appears in the text (EOT-style).
                # Uncovered text goes first, then the entry itself.
                if self._text:
                    out.append(self._make_filler(self._text))
                    self._text = ""
                out.append(record)
                self._entries.pop(0)
                continue
            if idx > 0:
                out.append(self._make_filler(self._text[:idx]))
            self._text = self._text[idx + len(token) :]
            out.append(record)
            self._entries.pop(0)
        if final and self._text:
            out.append(self._make_filler(self._text))
            self._text = ""
        return out


class _ChatStreamingMixin:
    # Composite-class attributes / cross-mixin methods; see the sibling
    # mixins for the same TYPE_CHECKING declaration pattern.
    if TYPE_CHECKING:
        provider: ProviderConfig
        model: str
        _client: httpx.Client
        _active_usage: usage_mod.UsageSink | None

        def _ensure_tokenizer(self) -> Tokenizer | None: ...
        def _intern(self, text: str) -> int: ...
        def _cands_from_list(self, items: list[dict]) -> list[TokenCandidate]: ...
        def _genstep_from_emit_record(
            self,
            record: tuple[int | None, str, float, Any, int | None],
            *,
            tokens_before: list[int],
            step_idx: int,
            is_last: bool,
            last_finish_reason: str | None,
            watch_ids: Sequence[int],
            note: str,
        ) -> GenStep: ...
        def _iter_completions_stream(
            self,
            body: dict[str, Any],
            *,
            extra_headers: dict[str, str] | None = None,
            path: str = "/completions",
        ) -> Iterator[dict[str, Any]]: ...

    def supports_chat_sampler(self, sampler_name: str) -> bool:
        """Can ``sampler_name`` run server-side on /chat/completions?"""
        return sampler_name in CHAT_NATIVE_SAMPLERS

    def stream_chat_native(
        self,
        messages: list[dict[str, Any]],
        *,
        sampler_name: str,
        sampler_params: dict[str, Any],
        max_tokens: int,
        top_k: int,
        seed: int = 0,
        tools: list[dict[str, Any]] | None = None,
        stop_texts: list[str] | None = None,
        watch_ids: Sequence[int] = (),
    ) -> Iterator[GenStep]:
        """Stream one assistant turn via ``/chat/completions`` SSE.

        ``messages`` / ``tools`` are OpenAI-shaped and forwarded verbatim
        -- the caller (the chat-mode UI) owns the conversation structure.
        ``logprobs: true`` + ``top_logprobs: N`` attach the per-token
        alternatives (NIM caps N at 20, OpenRouter at whatever the routed
        provider supports; ``require_parameters`` pins OpenRouter to
        providers that actually honour logprobs). Yielded ``GenStep``s
        reuse :meth:`_genstep_from_emit_record` so they are shape-
        identical to the /completions path -- including the one-token
        lookahead that stamps the terminal ``stop_reason``.

        Token ids: the chat schema carries token *text* only, no ids.
        When the per-model HF tokenizer is available (the
        ``[providers.*.tokenizers]`` mapping) we resolve each single-token
        text to its REAL model id so watch columns and ``tokens_before``
        stay meaningful; multi-token or unmapped texts fall back to
        synthetic interned ids, same as the legacy /completions parser.
        """
        if sampler_name not in CHAT_NATIVE_SAMPLERS:
            raise NotImplementedError(
                f"sampler {sampler_name!r} has no /chat/completions analogue "
                f"on {self.provider.name!r}; chat simulation mode supports "
                f"{sorted(CHAT_NATIVE_SAMPLERS)} only."
            )
        top = max(1, min(top_k, self.provider.max_top_logprobs))
        body: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": int(max_tokens),
            "stream": True,
            "stream_options": {"include_usage": True},
            "seed": int(seed),
            "logprobs": True,
            "top_logprobs": top,
        }
        body.update(_chat_sampler_params(sampler_name, sampler_params))
        if tools:
            body["tools"] = list(tools)
        if stop_texts:
            # Same 4-entry OpenAI cap as the /completions path.
            body["stop"] = [str(s) for s in stop_texts if s][:4]
        if self.provider.require_parameters:
            # OpenRouter: refuse to route to providers that would silently
            # drop logprobs/top_logprobs -- without this the stream "works"
            # but every distribution is empty.
            body.setdefault("provider", {})["require_parameters"] = True

        note_parts = [f"{sampler_name} (server-side, chat simulation)"]
        for knob in ("temperature", "top_p"):
            if knob in body and body[knob] is not None:
                note_parts.append(f"{knob}={body[knob]:g}")
        note = ", ".join(note_parts)

        # No meaningful local prompt ids: the provider renders the chat
        # template server-side, so the context starts opaque and grows
        # with each emitted token id.
        tokens_before: list[int] = []
        step_idx = 0
        prev_record: _EmitRecord | None = None
        last_finish_reason: str | None = None
        aligner = _ChatRecordAligner(self._nan_record)

        def _emit(rec: _EmitRecord) -> Iterator[GenStep]:
            # One-record lookahead so the terminal step can be stamped
            # with the mapped finish_reason (same idiom as /completions).
            nonlocal prev_record, step_idx
            if prev_record is not None:
                yield self._genstep_from_emit_record(
                    prev_record,
                    tokens_before=tokens_before,
                    step_idx=step_idx,
                    is_last=False,
                    last_finish_reason=last_finish_reason,
                    watch_ids=watch_ids,
                    note=note,
                )
                step_idx += 1
            prev_record = rec

        for chunk in self._iter_completions_stream(body, path="/chat/completions"):
            u = chunk.get("usage") if isinstance(chunk, dict) else None
            if isinstance(u, dict):
                usage_mod.record_tokens(
                    self._active_usage,
                    prompt_tokens=u.get("prompt_tokens"),
                    completion_tokens=u.get("completion_tokens"),
                    total_tokens=u.get("total_tokens"),
                )
            choices = chunk.get("choices") or []
            if not choices:
                continue
            ch = choices[0]
            fr = ch.get("finish_reason")
            if fr is not None:
                last_finish_reason = str(fr)
            delta_text, entries = self._chat_records(ch)
            for rec in aligner.push(delta_text, entries):
                yield from _emit(rec)

        for rec in aligner.flush():
            yield from _emit(rec)

        if prev_record is not None:
            yield self._genstep_from_emit_record(
                prev_record,
                tokens_before=tokens_before,
                step_idx=step_idx,
                is_last=True,
                last_finish_reason=last_finish_reason,
                watch_ids=watch_ids,
                note=note,
            )

    def _chat_records(self, choice: dict[str, Any]) -> tuple[str, list[tuple[str, _EmitRecord]]]:
        """Parse one streamed chat choice for the record aligner.

        Returns ``(delta_text, [(token_text, emit_record), ...])``. The
        chat schema nests per-token data under ``logprobs.content[]``
        with ``top_logprobs`` as a ``[{token, logprob}]`` list (which
        ``_cands_from_list`` / ``_candidates_from_top_entry`` already
        handle); the :class:`_ChatRecordAligner` then reconciles these
        entries with the ``delta.content`` text stream, because live
        providers were seen emitting the two out of lockstep.
        """
        records: list[tuple[str, _EmitRecord]] = []
        lp_obj = choice.get("logprobs") or {}
        entries = lp_obj.get("content") or []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            text = str(entry.get("token", ""))
            lp_raw = entry.get("logprob")
            lp = float(lp_raw) if lp_raw is not None else float("nan")
            tops = entry.get("top_logprobs") or []
            # Inject resolved token ids into the top-logprob items so the
            # shared ``_cands_from_list`` parser emits real model ids for
            # the alternatives table (and watch columns can match them).
            for item in tops:
                if isinstance(item, dict) and "token_id" not in item:
                    item["token_id"] = self._resolve_chat_token_id(str(item.get("token", "")))
            records.append((text, (self._resolve_chat_token_id(text), text, lp, tops, None)))
        delta = choice.get("delta") or {}
        return str(delta.get("content") or ""), records

    def _nan_record(self, text: str) -> _EmitRecord:
        """Filler record for delta text that has no logprob entry."""
        return (self._resolve_chat_token_id(text), text, float("nan"), [], None)

    def _resolve_chat_token_id(self, text: str) -> int | None:
        """Map a chat-stream token text to a real model id when possible.

        Single-token texts under the local HF tokenizer resolve to the
        model's actual id (making watch columns / ``tokens_before``
        meaningful); anything else falls back to the synthetic intern
        space. Returning the id (never ``None`` in practice) keeps the
        record shape compatible with the shared emit-record builder.
        """
        tok = self._ensure_tokenizer()
        if tok is not None and text:
            try:
                ids = tok.encode(text, add_special_tokens=False).ids
            except Exception:
                ids = []
            if len(ids) == 1:
                return int(ids[0])
        return self._intern(text)


def _chat_sampler_params(name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Translate a builtin sampler spec to /chat/completions body params."""
    if name == "greedy":
        return {"temperature": 0.0}
    if name == "temperature":
        return {"temperature": float(params.get("t", params.get("temperature", 1.0)))}
    # top_p: nucleus mass + the temperature applied before truncation.
    return {
        "temperature": float(params.get("t", params.get("temperature", 1.0))),
        "top_p": float(params.get("p", params.get("top_p", 1.0))),
    }
