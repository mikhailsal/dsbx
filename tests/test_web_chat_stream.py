"""Route-guard + SSE tests for the chat-simulation generate path.

Exercises ``POST /api/v1/generate/stream`` with ``messages[]`` through
the real FastAPI app: the full guard matrix (chat backend x messages
presence), sampler gating, and the ``step* -> usage -> done`` frame
sequence produced by :mod:`dsbx.web.streaming_chat`.
"""

from __future__ import annotations

import json
import math

from dsbx.core.engine import GenStep
from dsbx.core.samplers import SamplerDecision
from dsbx.core.types import StepResult, TokenCandidate
from tests.fakes import FakeBackend
from tests.web_helpers import build_test_app, make_authed_client


class ChatFakeBackend(FakeBackend):
    """Chat-only fake: refuses text continuation, streams canned chat steps."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("generation_disabled", True)
        kwargs.setdefault("supports_chat_stream", True)
        super().__init__(**kwargs)
        self.chat_calls: list[dict] = []
        self.chat_error: Exception | None = None

    def supports_chat_sampler(self, sampler_name: str) -> bool:
        return sampler_name in {"greedy", "temperature", "top_p"}

    def stream_chat_native(self, messages, **kwargs):
        self.chat_calls.append({"messages": messages, **kwargs})
        if self.chat_error is not None:
            raise self.chat_error
        for i, (tid, text, lp) in enumerate([(500, " Hi", -0.1), (501, "!", -0.4)]):
            cand = TokenCandidate(tid, text, lp, 0)
            yield GenStep(
                step=i,
                tokens_before=[500] * i,
                step_result=StepResult(
                    position=i, candidates=[cand], is_full_vocab=False, chosen=cand
                ),
                decision=SamplerDecision(
                    token_id=tid, token_text=text, kept=[], greedy_token_id=tid
                ),
                stop_reason="user_stop" if i == 1 else None,
            )


def _parse_sse(body: str) -> list[dict]:
    events = []
    for line in body.splitlines():
        if line.startswith("data:"):
            events.append(json.loads(line[5:].strip()))
    return events


def _request(client, backend="chatprov", messages=..., sampler=None, **extra):
    if messages is ...:
        messages = [{"role": "user", "content": "hi"}]
    payload = {
        "backend": backend,
        "prompt": "",
        "sampler": sampler or {"name": "greedy", "params": {}},
        "max_tokens": 4,
        "top_k": 5,
        **extra,
    }
    if messages is not None:
        payload["messages"] = messages
    return client.post("/api/v1/generate/stream", json=payload)


def _make_client(chat_backend=None, text_backend=None):
    backends = {}
    backends["chatprov"] = chat_backend or ChatFakeBackend(name="chatprov")
    backends["dsbx-host-py"] = text_backend or FakeBackend(name="host")
    app = build_test_app(backends)
    return make_authed_client(app), backends


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #
def test_chat_messages_stream_emits_step_usage_done() -> None:
    client, backends = _make_client()
    r = _request(client)
    assert r.status_code == 200
    events = _parse_sse(r.text)
    kinds = [e["event"] for e in events]
    assert kinds == ["step", "step", "usage", "done"]
    assert events[0]["step"]["decision"]["token_id"] == 500
    assert events[-1]["stop_reason"] == "user_stop"
    assert events[-1].get("error") is None
    # completion_tokens falls back to the emitted step count.
    assert events[2]["completion_tokens"] == 2
    # Messages arrived verbatim at the backend.
    call = backends["chatprov"].chat_calls[0]
    assert call["messages"] == [{"role": "user", "content": "hi"}]
    assert call["max_tokens"] == 4


def test_chat_stream_forwards_tools_and_stop_texts() -> None:
    client, backends = _make_client()
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    r = _request(client, tools=tools, stop_texts=["END"])
    assert r.status_code == 200
    call = backends["chatprov"].chat_calls[0]
    assert call["tools"] == tools
    assert call["stop_texts"] == ["END"]


def test_chat_stream_resolves_watches() -> None:
    chat = ChatFakeBackend(name="chatprov", tokens={"Hi": [500]}, eos_token_ids=(99,))
    client, _ = _make_client(chat_backend=chat)
    r = _request(client, watch_texts=["Hi"], watch_ids=[7], watch_eos=True)
    assert r.status_code == 200
    assert chat.chat_calls[0]["watch_ids"] == [500, 7, 99]


def test_chat_stream_error_lands_in_done_frame() -> None:
    chat = ChatFakeBackend(name="chatprov")
    chat.chat_error = RuntimeError("upstream exploded")
    client, _ = _make_client(chat_backend=chat)
    r = _request(client)
    assert r.status_code == 200  # headers already committed; error rides `done`
    events = _parse_sse(r.text)
    assert events[-1]["event"] == "done"
    assert "upstream exploded" in events[-1]["error"]


# --------------------------------------------------------------------------- #
# Guard matrix
# --------------------------------------------------------------------------- #
def test_messages_against_template_capable_backend_400() -> None:
    client, _ = _make_client()
    r = _request(client, backend="dsbx-host-py")
    assert r.status_code == 400
    assert "render the chat template client-side" in r.json()["detail"]


def test_plain_prompt_against_chat_only_backend_400() -> None:
    client, _ = _make_client()
    r = _request(client, messages=None, prompt="raw text")
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "chat-only" in detail
    assert "chat mode" in detail


def test_empty_messages_list_400() -> None:
    client, _ = _make_client()
    r = _request(client, messages=[])
    assert r.status_code == 400
    assert "non-empty" in r.json()["detail"]


def test_unsupported_sampler_in_chat_mode_400() -> None:
    client, _ = _make_client()
    r = _request(client, sampler={"name": "min_p", "params": {"p": 0.1}})
    assert r.status_code == 400
    assert "simulation mode" in r.json()["detail"]


def test_prompt_path_still_works_without_messages() -> None:
    """Regression: the messages field must not disturb the prompt path."""
    text_backend = FakeBackend(
        name="host",
        tokens={"ab": [97, 98]},
        pieces={97: "a", 98: "b", 88: "X"},
        distributions={
            (97, 98): [TokenCandidate(88, "X", math.log(0.9), 0)],
            (97, 98, 88): [TokenCandidate(88, "X", math.log(0.9), 0)],
        },
    )
    client, _ = _make_client(text_backend=text_backend)
    r = _request(client, backend="dsbx-host-py", messages=None, prompt="ab", max_tokens=1)
    assert r.status_code == 200
    events = _parse_sse(r.text)
    assert [e["event"] for e in events][-1] == "done"
    assert any(e["event"] == "step" for e in events)
