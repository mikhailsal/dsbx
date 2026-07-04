"""Tests for the /chat/completions streaming path (chat simulation mode).

Backend layer: ``OpenAICompatBackend.stream_chat_native`` against a
mocked SSE stream -- request-body shape (messages / logprobs /
require_parameters / tools / stop), sampler translation, chunk parsing
into :class:`GenStep`, token-id resolution through the local tokenizer,
finish_reason mapping, and usage accounting.
"""

from __future__ import annotations

import json
from contextlib import contextmanager

import pytest

import dsbx.backends.openai_compat.backend as oc_mod
from dsbx.backends.openai_compat import OpenAICompatBackend
from dsbx.core import usage as usage_mod
from dsbx.core.config import ProviderConfig
from tests.fakes import MockHTTPClient


# --------------------------------------------------------------------------- #
# Harness (mirrors test_backends_http.py, trimmed to the chat-only shape)
# --------------------------------------------------------------------------- #
class _FakeTok:
    """Single-token resolver: maps known words to ids, splits the rest."""

    def __init__(self, word_to_id: dict[str, int]) -> None:
        self._map = word_to_id

    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens

        class _Enc:
            def __init__(self, ids: list[int]) -> None:
                self.ids = ids

        if text in self._map:
            return _Enc([self._map[text]])
        return _Enc([hash(ch) % 1000 for ch in text])


def _make_chat_backend(
    monkeypatch: pytest.MonkeyPatch,
    *,
    require_parameters: bool = False,
    with_tokenizer: bool = True,
    max_top: int = 20,
) -> tuple[OpenAICompatBackend, MockHTTPClient]:
    mock = MockHTTPClient({})
    monkeypatch.setattr(oc_mod.httpx, "Client", lambda **kw: mock)
    prov = ProviderConfig(
        name="nimtest",
        base_url="https://api.test/v1",
        api_key_env="TEST_API_KEY",
        default_model="test/chat-model",
        max_top_logprobs=max_top,
        require_parameters=require_parameters,
        has_completions=False,
    )
    backend = OpenAICompatBackend(prov, model="test/chat-model", max_retries=0)
    if with_tokenizer:
        fake = _FakeTok({" Paris": 500, " London": 501, "Hello": 502})
        monkeypatch.setattr(backend, "_ensure_tokenizer", lambda: fake)
    else:
        monkeypatch.setattr(backend, "_ensure_tokenizer", lambda: None)
    return backend, mock


def _sse_lines(chunks: list[dict]) -> list[bytes]:
    out: list[bytes] = []
    for ch in chunks:
        out.append(f"data: {json.dumps(ch)}".encode())
        out.append(b"")
    out.append(b"data: [DONE]")
    return out


class _MockStreamResponse:
    def __init__(self, status_code: int, lines: list[bytes]) -> None:
        self.status_code = status_code
        self.headers: dict = {}
        self._lines = lines

    def iter_lines(self):
        yield from self._lines

    def raise_for_status(self):
        if not (200 <= self.status_code < 300):
            raise RuntimeError(f"HTTP {self.status_code}")


def _attach_stream(mock: MockHTTPClient, chunks: list[dict]) -> None:
    resp = _MockStreamResponse(200, _sse_lines(chunks))

    @contextmanager
    def _stream(method, url, **kwargs):
        mock.calls.append({"method": method, "url": url, "kwargs": kwargs, "stream": True})
        yield resp

    mock.stream = _stream  # type: ignore[attr-defined]


def _chat_chunk(token: str, logprob: float, tops: list[tuple[str, float]], finish=None) -> dict:
    return {
        "choices": [
            {
                "delta": {"content": token},
                "logprobs": {
                    "content": [
                        {
                            "token": token,
                            "logprob": logprob,
                            "top_logprobs": [{"token": t, "logprob": lp} for t, lp in tops],
                        }
                    ]
                },
                "finish_reason": finish,
            }
        ]
    }


_MESSAGES = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Capital of France?"},
]


# --------------------------------------------------------------------------- #
# Request body shape
# --------------------------------------------------------------------------- #
def test_chat_stream_body_shape_and_path(monkeypatch) -> None:
    backend, mock = _make_chat_backend(monkeypatch)
    _attach_stream(mock, [_chat_chunk(" Paris", -0.1, [(" Paris", -0.1)], finish="stop")])
    steps = list(
        backend.stream_chat_native(
            _MESSAGES,
            sampler_name="temperature",
            sampler_params={"t": 0.7},
            max_tokens=8,
            top_k=5,
            seed=42,
        )
    )
    call = mock.calls[-1]
    assert call["url"] == "/chat/completions"
    body = call["kwargs"]["json"]
    assert body["messages"] == _MESSAGES
    assert body["logprobs"] is True
    assert body["top_logprobs"] == 5
    assert body["stream"] is True
    assert body["temperature"] == pytest.approx(0.7)
    assert body["seed"] == 42
    assert "provider" not in body
    assert len(steps) == 1


def test_chat_stream_top_logprobs_capped_by_provider(monkeypatch) -> None:
    backend, mock = _make_chat_backend(monkeypatch, max_top=20)
    _attach_stream(mock, [_chat_chunk("x", -0.1, [])])
    list(
        backend.stream_chat_native(
            _MESSAGES, sampler_name="greedy", sampler_params={}, max_tokens=1, top_k=50
        )
    )
    assert mock.calls[-1]["kwargs"]["json"]["top_logprobs"] == 20


def test_chat_stream_openrouter_require_parameters(monkeypatch) -> None:
    backend, mock = _make_chat_backend(monkeypatch, require_parameters=True)
    _attach_stream(mock, [_chat_chunk("x", -0.1, [])])
    list(
        backend.stream_chat_native(
            _MESSAGES, sampler_name="greedy", sampler_params={}, max_tokens=1, top_k=5
        )
    )
    body = mock.calls[-1]["kwargs"]["json"]
    assert body["provider"] == {"require_parameters": True}


def test_chat_stream_tools_and_stop_forwarded(monkeypatch) -> None:
    backend, mock = _make_chat_backend(monkeypatch)
    _attach_stream(mock, [_chat_chunk("x", -0.1, [])])
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    list(
        backend.stream_chat_native(
            _MESSAGES,
            sampler_name="greedy",
            sampler_params={},
            max_tokens=1,
            top_k=5,
            tools=tools,
            stop_texts=["\n\n", "END", "a", "b", "c"],  # 5 -> capped at 4
        )
    )
    body = mock.calls[-1]["kwargs"]["json"]
    assert body["tools"] == tools
    assert body["stop"] == ["\n\n", "END", "a", "b"]


# --------------------------------------------------------------------------- #
# Sampler translation
# --------------------------------------------------------------------------- #
def test_chat_sampler_translation_matrix(monkeypatch) -> None:
    backend, mock = _make_chat_backend(monkeypatch)
    cases = [
        ("greedy", {}, {"temperature": 0.0}),
        ("temperature", {"t": 1.3}, {"temperature": 1.3}),
        ("top_p", {"p": 0.9, "t": 0.8}, {"temperature": 0.8, "top_p": 0.9}),
    ]
    for name, params, expected in cases:
        _attach_stream(mock, [_chat_chunk("x", -0.1, [])])
        list(
            backend.stream_chat_native(
                _MESSAGES, sampler_name=name, sampler_params=params, max_tokens=1, top_k=5
            )
        )
        body = mock.calls[-1]["kwargs"]["json"]
        for key, val in expected.items():
            assert body[key] == pytest.approx(val), (name, key)


def test_chat_stream_rejects_unsupported_sampler(monkeypatch) -> None:
    backend, mock = _make_chat_backend(monkeypatch)
    with pytest.raises(NotImplementedError, match="no /chat/completions analogue"):
        list(
            backend.stream_chat_native(
                _MESSAGES, sampler_name="min_p", sampler_params={}, max_tokens=1, top_k=5
            )
        )
    assert mock.calls == []  # short-circuits before any I/O


def test_supports_chat_sampler_matrix(monkeypatch) -> None:
    backend, _ = _make_chat_backend(monkeypatch)
    assert backend.supports_chat_sampler("greedy")
    assert backend.supports_chat_sampler("temperature")
    assert backend.supports_chat_sampler("top_p")
    assert not backend.supports_chat_sampler("top_k")
    assert not backend.supports_chat_sampler("mirostat")
    assert not backend.supports_chat_sampler("custom")


# --------------------------------------------------------------------------- #
# Chunk parsing -> GenStep
# --------------------------------------------------------------------------- #
def test_chat_stream_parses_steps_with_real_token_ids(monkeypatch) -> None:
    """Single-token texts resolve to REAL ids via the local tokenizer."""
    backend, mock = _make_chat_backend(monkeypatch)
    _attach_stream(
        mock,
        [
            _chat_chunk(" Paris", -0.2, [(" Paris", -0.2), (" London", -1.5)]),
            _chat_chunk("!", -0.5, [("!", -0.5)], finish="stop"),
        ],
    )
    steps = list(
        backend.stream_chat_native(
            _MESSAGES, sampler_name="greedy", sampler_params={}, max_tokens=4, top_k=5
        )
    )
    assert len(steps) == 2
    first = steps[0]
    # " Paris" resolved through the fake tokenizer to its real id.
    assert first.decision.token_id == 500
    assert first.step_result.chosen is not None
    assert first.step_result.chosen.token_id == 500
    # The runner-up " London" also carries its real id in the table.
    alt_ids = [c.token_id for c in first.step_result.candidates]
    assert 501 in alt_ids
    assert first.stop_reason is None
    # tokens_before grows with emitted ids (starts empty: the provider
    # renders the template server-side, we never see prompt ids).
    assert steps[1].tokens_before == [500]
    # Terminal step carries the mapped finish_reason.
    assert steps[1].stop_reason == "user_stop"


def test_chat_stream_finish_reason_length(monkeypatch) -> None:
    backend, mock = _make_chat_backend(monkeypatch)
    _attach_stream(mock, [_chat_chunk("Hello", -0.3, [("Hello", -0.3)], finish="length")])
    steps = list(
        backend.stream_chat_native(
            _MESSAGES, sampler_name="greedy", sampler_params={}, max_tokens=1, top_k=5
        )
    )
    assert steps[-1].stop_reason == "max_tokens"


def test_chat_stream_interns_without_tokenizer(monkeypatch) -> None:
    """No local tokenizer -> synthetic intern ids, stream still works."""
    backend, mock = _make_chat_backend(monkeypatch, with_tokenizer=False)
    _attach_stream(mock, [_chat_chunk(" Paris", -0.2, [(" Paris", -0.2)], finish="stop")])
    steps = list(
        backend.stream_chat_native(
            _MESSAGES, sampler_name="greedy", sampler_params={}, max_tokens=1, top_k=5
        )
    )
    assert len(steps) == 1
    assert steps[0].decision.token_id >= backend._INTERN_ID_BASE
    assert steps[0].decision.token_text == " Paris"


def test_chat_stream_delta_without_logprobs_degrades_to_nan(monkeypatch) -> None:
    """Provider quirk: content delta with no logprobs must still stream."""
    backend, mock = _make_chat_backend(monkeypatch)
    chunk = {
        "choices": [{"delta": {"content": "Hello"}, "logprobs": None, "finish_reason": "stop"}]
    }
    _attach_stream(mock, [chunk])
    steps = list(
        backend.stream_chat_native(
            _MESSAGES, sampler_name="greedy", sampler_params={}, max_tokens=1, top_k=5
        )
    )
    assert len(steps) == 1
    assert steps[0].decision.token_text == "Hello"
    assert steps[0].step_result.candidates == []


def test_chat_stream_whole_completion_in_one_delta(monkeypatch) -> None:
    """Live-seen OpenRouter (WandB route) shape: the ENTIRE completion
    arrives as one ``delta.content`` while ``logprobs.content`` carries
    an entry only for the final EOT token. The user's text must stream
    (as a NaN-logprob record) BEFORE the logprob-backed EOT step."""
    backend, mock = _make_chat_backend(monkeypatch)
    chunk = {
        "choices": [
            {
                "delta": {"content": "Paris."},
                "logprobs": {
                    "content": [
                        {
                            "token": "<|eot_id|>",
                            "logprob": -0.01,
                            "top_logprobs": [{"token": "<|eot_id|>", "logprob": -0.01}],
                        }
                    ]
                },
                "finish_reason": "stop",
            }
        ]
    }
    _attach_stream(mock, [chunk])
    steps = list(
        backend.stream_chat_native(
            _MESSAGES, sampler_name="greedy", sampler_params={}, max_tokens=8, top_k=5
        )
    )
    assert [s.decision.token_text for s in steps] == ["Paris.", "<|eot_id|>"]
    assert steps[0].step_result.candidates == []  # NaN degraded, no fake alts
    assert steps[1].step_result.candidates  # the EOT entry kept its logprobs
    assert steps[-1].stop_reason == "user_stop"


def test_chat_stream_offset_logprob_entries_realign(monkeypatch) -> None:
    """Live-seen OpenRouter/vLLM shape: logprob entries lag the delta
    text by several tokens (chunk 1 says text "The cap" but carries the
    entry for " France"). The aligner must emit every character of the
    text exactly once, attaching each entry at its match position and
    filling unmatched text with NaN records."""
    backend, mock = _make_chat_backend(monkeypatch)

    def chunk(text: str, entry_token: str | None, finish=None) -> dict:
        lp = None
        if entry_token is not None:
            lp = {"content": [{"token": entry_token, "logprob": -0.05, "top_logprobs": []}]}
        return {"choices": [{"delta": {"content": text}, "logprobs": lp, "finish_reason": finish}]}

    _attach_stream(
        mock,
        [
            chunk("The cap", " France"),
            chunk("ita", " is"),
            chunk("l of F", " Paris"),
            chunk("r", "."),
            chunk("ance is Paris.", "<|eot_id|>", finish="stop"),
        ],
    )
    steps = list(
        backend.stream_chat_native(
            _MESSAGES, sampler_name="greedy", sampler_params={}, max_tokens=16, top_k=5
        )
    )
    texts = [s.decision.token_text for s in steps]
    assert texts == ["The capital of", " France", " is", " Paris", ".", "<|eot_id|>"]
    # Reassembled text == exactly what the provider streamed (no dupes).
    assert "".join(texts) == "The capital of France is Paris.<|eot_id|>"
    assert steps[-1].stop_reason == "user_stop"


def test_chat_stream_delta_covered_by_entries_not_duplicated(monkeypatch) -> None:
    """The well-behaved shape (delta text == the logprob entries' text)
    must NOT grow an extra NaN record."""
    backend, mock = _make_chat_backend(monkeypatch)
    _attach_stream(mock, [_chat_chunk(" Paris", -0.2, [(" Paris", -0.2)], finish="stop")])
    steps = list(
        backend.stream_chat_native(
            _MESSAGES, sampler_name="greedy", sampler_params={}, max_tokens=2, top_k=5
        )
    )
    assert [s.decision.token_text for s in steps] == [" Paris"]


def test_chat_stream_records_usage(monkeypatch) -> None:
    backend, mock = _make_chat_backend(monkeypatch)
    usage_chunk = {
        "choices": [],
        "usage": {"prompt_tokens": 25, "completion_tokens": 2, "total_tokens": 27},
    }
    _attach_stream(
        mock,
        [_chat_chunk(" Paris", -0.2, [], finish="stop"), usage_chunk],
    )
    sink = usage_mod.make_sink()
    backend.set_active_usage(sink)
    try:
        list(
            backend.stream_chat_native(
                _MESSAGES, sampler_name="greedy", sampler_params={}, max_tokens=2, top_k=5
            )
        )
    finally:
        backend.set_active_usage(None)
    assert sink["prompt_tokens"] == 25
    assert sink["completion_tokens"] == 2
    assert sink["total_tokens"] == 27
    assert sink["requests"] == 1


# --------------------------------------------------------------------------- #
# Capabilities
# --------------------------------------------------------------------------- #
def test_chat_only_capabilities_advertise_chat_stream(monkeypatch) -> None:
    backend, _ = _make_chat_backend(monkeypatch)
    caps = backend.capabilities
    assert caps.generation_disabled is True
    assert caps.supports_chat_stream is True
    assert "chat mode" in caps.notes


def test_completions_capable_backend_does_not_advertise_chat_stream(monkeypatch) -> None:
    mock = MockHTTPClient({})
    monkeypatch.setattr(oc_mod.httpx, "Client", lambda **kw: mock)
    prov = ProviderConfig(
        name="fw",
        base_url="https://api.test/v1",
        api_key_env="TEST_API_KEY",
        default_model="m",
        has_completions=True,
    )
    backend = OpenAICompatBackend(prov, model="m", max_retries=0)
    caps = backend.capabilities
    assert caps.generation_disabled is False
    assert caps.supports_chat_stream is False
