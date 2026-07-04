"""Chat-template discovery tests: core helpers + per-backend implementations.

Covers the whole discovery matrix from the plan:

- core normalization helpers (``template_from_config_value``, ``token_text``,
  ``info_from_tokenizer_config``) across the shapes HF repos actually ship;
- ``fetch_hf_chat_template`` against a mocked ``hf_hub_download`` (success,
  standalone ``chat_template.jinja`` fallback, network failure -> ``note``);
- ``OpenAICompatBackend.chat_template_info`` (mapped repo, unmapped model,
  once-only caching);
- ``HFBackend`` / ``LlamaCppPyBackend`` extraction from their tokenizers
  (via ``__new__`` object stubs -- no torch / llama.cpp needed);
- ``RemoteBackend`` proxying of ``GET /v1/chat_template`` including the
  older-server 404 degradation and cache invalidation on refresh;
- the dsbx-serve route itself (mounted by ``add_chat_template_route``).
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from dsbx.backends.hf import HFBackend
from dsbx.backends.llamacpp_py import LlamaCppPyBackend
from dsbx.backends.openai_compat import OpenAICompatBackend
from dsbx.backends.remote import RemoteBackend
from dsbx.core import chat_template as ct
from dsbx.core.config import ProviderConfig
from dsbx.server.app import make_app
from tests.fakes import FakeBackend, MockHTTPClient

CHATML = "{% for message in messages %}...{% endfor %}"


# --------------------------------------------------------------------------- #
# Core helpers
# --------------------------------------------------------------------------- #
def test_template_from_config_value_shapes() -> None:
    assert ct.template_from_config_value(CHATML) == CHATML
    assert ct.template_from_config_value("") is None
    assert ct.template_from_config_value(None) is None
    # Multi-template list: prefer the "default" entry.
    lst = [
        {"name": "tool_use", "template": "TOOLS"},
        {"name": "default", "template": "DEFAULT"},
    ]
    assert ct.template_from_config_value(lst) == "DEFAULT"
    # No "default" name -> first entry wins.
    assert ct.template_from_config_value([{"name": "only", "template": "X"}]) == "X"
    # Older transformers dict shape.
    assert ct.template_from_config_value({"default": "D", "other": "O"}) == "D"
    assert ct.template_from_config_value({"other": "O"}) == "O"
    assert ct.template_from_config_value([]) is None


def test_token_text_shapes() -> None:
    assert ct.token_text("<s>") == "<s>"
    assert ct.token_text({"content": "<|eot_id|>", "special": True}) == "<|eot_id|>"
    assert ct.token_text(None) is None
    assert ct.token_text({"no_content": 1}) is None
    assert ct.token_text(42) is None


def test_info_from_tokenizer_config_extracts_specials_and_template() -> None:
    cfg = {
        "chat_template": CHATML,
        "bos_token": "<|begin_of_text|>",
        "eos_token": {"content": "<|eot_id|>"},
        "pad_token": None,
    }
    info = ct.info_from_tokenizer_config(cfg, source="hf_hub")
    assert info.template == CHATML
    assert info.source == "hf_hub"
    assert info.bos_token == "<|begin_of_text|>"
    assert info.eos_token == "<|eot_id|>"
    assert info.special_tokens == {
        "bos_token": "<|begin_of_text|>",
        "eos_token": "<|eot_id|>",
    }
    assert info.is_base_model is False


def test_info_from_tokenizer_config_base_model() -> None:
    info = ct.info_from_tokenizer_config({"eos_token": "</s>"}, source="hf_hub")
    assert info.template is None
    assert info.source == "none"
    assert info.is_base_model is True


def test_template_override_wins_over_config() -> None:
    cfg = {"chat_template": "FROM_CONFIG"}
    info = ct.info_from_tokenizer_config(cfg, source="hf_hub", template_override="FROM_FILE")
    assert info.template == "FROM_FILE"


def test_is_base_model_false_when_discovery_failed() -> None:
    degraded = ct.none_info(note="could not fetch tokenizer_config.json (HTTPError)")
    assert degraded.template is None
    assert degraded.is_base_model is False  # failure, not proof of base model


def test_to_dict_from_dict_round_trip() -> None:
    info = ct.ChatTemplateInfo(
        template=CHATML,
        source="gguf",
        bos_token="<s>",
        eos_token="</s>",
        special_tokens={"bos_token": "<s>", "eos_token": "</s>"},
        note="",
    )
    restored = ct.ChatTemplateInfo.from_dict(info.to_dict())
    assert restored == info


# --------------------------------------------------------------------------- #
# fetch_hf_chat_template (mocked hub)
# --------------------------------------------------------------------------- #
def _write_hub_files(tmp_path, monkeypatch, files: dict[str, str], missing: set[str] = frozenset()):
    """Patch ``hf_hub_download`` to serve ``files`` from tmp_path."""

    def fake_download(repo_id: str, filename: str) -> str:
        if filename in missing or filename not in files:
            raise FileNotFoundError(f"{repo_id}/{filename} not found")
        p = tmp_path / filename
        p.write_text(files[filename], encoding="utf-8")
        return str(p)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)


def test_fetch_hf_chat_template_from_tokenizer_config(tmp_path, monkeypatch) -> None:
    _write_hub_files(
        tmp_path,
        monkeypatch,
        {"tokenizer_config.json": json.dumps({"chat_template": CHATML, "eos_token": "</s>"})},
    )
    info = ct.fetch_hf_chat_template("org/model")
    assert info.template == CHATML
    assert info.source == "hf_hub"
    assert info.eos_token == "</s>"


def test_fetch_hf_chat_template_standalone_jinja_fallback(tmp_path, monkeypatch) -> None:
    _write_hub_files(
        tmp_path,
        monkeypatch,
        {
            "tokenizer_config.json": json.dumps({"bos_token": "<s>"}),
            "chat_template.jinja": "JINJA_FILE_TEMPLATE",
        },
    )
    info = ct.fetch_hf_chat_template("org/model")
    assert info.template == "JINJA_FILE_TEMPLATE"
    assert info.bos_token == "<s>"


def test_fetch_hf_chat_template_base_model(tmp_path, monkeypatch) -> None:
    _write_hub_files(
        tmp_path,
        monkeypatch,
        {"tokenizer_config.json": json.dumps({"eos_token": "<|endoftext|>"})},
        missing={"chat_template.jinja"},
    )
    info = ct.fetch_hf_chat_template("org/base-model")
    assert info.template is None
    assert info.is_base_model is True


def test_fetch_hf_chat_template_network_failure(tmp_path, monkeypatch) -> None:
    _write_hub_files(tmp_path, monkeypatch, {}, missing={"tokenizer_config.json"})
    info = ct.fetch_hf_chat_template("org/gated-model")
    assert info.template is None
    assert "could not fetch tokenizer_config.json" in info.note
    assert info.is_base_model is False


# --------------------------------------------------------------------------- #
# Backend default + FakeBackend
# --------------------------------------------------------------------------- #
def test_backend_default_returns_none_info() -> None:
    assert FakeBackend().chat_template_info() == ct.none_info()


# --------------------------------------------------------------------------- #
# OpenAICompatBackend
# --------------------------------------------------------------------------- #
def _provider(**overrides) -> ProviderConfig:
    defaults: dict = {
        "name": "testprov",
        "base_url": "https://api.example/v1",
        "api_key_env": "TEST_KEY",
        "default_model": "test/model",
        "tokenizers": {"test/model": "org/hf-model"},
    }
    defaults.update(overrides)
    return ProviderConfig(**defaults)


def test_openai_compat_fetches_and_caches(monkeypatch) -> None:
    calls: list[str] = []

    def fake_fetch(repo: str) -> ct.ChatTemplateInfo:
        calls.append(repo)
        return ct.ChatTemplateInfo(template=CHATML, source="hf_hub")

    monkeypatch.setattr("dsbx.backends.openai_compat._tokenizer.fetch_hf_chat_template", fake_fetch)
    backend = OpenAICompatBackend(_provider())
    backend._client = MockHTTPClient({})  # never used here; keeps close() safe

    info = backend.chat_template_info()
    assert info.template == CHATML
    assert calls == ["org/hf-model"]
    # Second call must hit the cache, not the hub.
    assert backend.chat_template_info() is info
    assert calls == ["org/hf-model"]


def test_openai_compat_unmapped_model_degrades() -> None:
    backend = OpenAICompatBackend(_provider(tokenizers={}))
    backend._client = MockHTTPClient({})
    info = backend.chat_template_info()
    assert info.template is None
    assert "no HF tokenizer repo mapped" in info.note
    assert info.is_base_model is False


# --------------------------------------------------------------------------- #
# HFBackend (object stub -- no torch)
# --------------------------------------------------------------------------- #
class _StubHFTokenizer:
    chat_template = CHATML
    bos_token = "<|begin_of_text|>"
    eos_token = "<|eot_id|>"
    unk_token = None
    pad_token = "<|pad|>"


def test_hf_backend_reads_transformers_tokenizer() -> None:
    backend = HFBackend.__new__(HFBackend)
    backend.tokenizer = _StubHFTokenizer()
    info = backend.chat_template_info()
    assert info.template == CHATML
    assert info.source == "transformers"
    assert info.bos_token == "<|begin_of_text|>"
    assert info.eos_token == "<|eot_id|>"
    assert info.special_tokens["pad_token"] == "<|pad|>"
    assert "unk_token" not in info.special_tokens


def test_hf_backend_base_model() -> None:
    backend = HFBackend.__new__(HFBackend)
    tok = _StubHFTokenizer()
    tok.chat_template = None
    backend.tokenizer = tok
    info = backend.chat_template_info()
    assert info.template is None
    assert info.is_base_model is True
    assert info.eos_token == "<|eot_id|>"


# --------------------------------------------------------------------------- #
# LlamaCppPyBackend (object stub -- no llama.cpp)
# --------------------------------------------------------------------------- #
def _llamacpp_stub(metadata: dict, pieces: dict[int, str]) -> LlamaCppPyBackend:
    backend = LlamaCppPyBackend.__new__(LlamaCppPyBackend)

    class _StubLlama:
        def __init__(self) -> None:
            self.metadata = metadata

        def detokenize(self, ids, special=False):
            del special
            return "".join(pieces.get(i, "") for i in ids).encode("utf-8")

    backend._llama = _StubLlama()
    backend._piece_cache = {}
    backend._bos_ids = (1,)
    backend._eos_ids = (2,)
    return backend


def test_llamacpp_py_reads_gguf_metadata() -> None:
    backend = _llamacpp_stub(
        {"tokenizer.chat_template": CHATML},
        {1: "<|im_start|>", 2: "<|im_end|>"},
    )
    info = backend.chat_template_info()
    assert info.template == CHATML
    assert info.source == "gguf"
    assert info.bos_token == "<|im_start|>"
    assert info.eos_token == "<|im_end|>"


def test_llamacpp_py_base_model_gguf() -> None:
    backend = _llamacpp_stub({}, {1: "<s>", 2: "</s>"})
    info = backend.chat_template_info()
    assert info.template is None
    assert info.is_base_model is True


# --------------------------------------------------------------------------- #
# dsbx-serve route + RemoteBackend proxy
# --------------------------------------------------------------------------- #
def test_server_route_returns_template() -> None:
    fake = FakeBackend(
        chat_template=ct.ChatTemplateInfo(
            template=CHATML, source="gguf", bos_token="<s>", eos_token="</s>"
        )
    )
    with TestClient(make_app(fake, backend_kind="fake-kind")) as client:
        r = client.get("/v1/chat_template")
    assert r.status_code == 200
    data = r.json()
    assert data["template"] == CHATML
    assert data["source"] == "gguf"
    assert data["is_base_model"] is False


def test_server_route_base_model() -> None:
    with TestClient(make_app(FakeBackend(), backend_kind="fake-kind")) as client:
        r = client.get("/v1/chat_template")
    assert r.status_code == 200
    assert r.json()["template"] is None
    assert r.json()["is_base_model"] is True


def test_remote_backend_proxies_chat_template() -> None:
    fake = FakeBackend(chat_template=ct.ChatTemplateInfo(template=CHATML, source="gguf"))
    app = make_app(fake, backend_kind="fake-kind")
    remote = RemoteBackend("http://testserver", client=TestClient(app))
    info = remote.chat_template_info()
    assert info.template == CHATML
    assert info.source == "remote"  # remote proxy re-labels the origin
    # Cached: swapping the upstream template does not change the handle...
    fake._chat_template = None
    assert remote.chat_template_info() is info
    # ...until refresh_info() invalidates the cache (model swap path).
    remote.refresh_info()
    assert remote.chat_template_info().template is None


def test_remote_backend_degrades_on_old_server() -> None:
    """A dsbx-serve without /v1/chat_template (404) must not look like a base model."""
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/info":
            return httpx.Response(
                200,
                json={
                    "capabilities": None,
                    "engine_version": "0.0.1",
                    "backend_kind": "old",
                    "loaded_model": None,
                    "state": "ready",
                },
            )
        return httpx.Response(404, json={"detail": "not found"})

    remote = RemoteBackend("http://testserver", transport=httpx.MockTransport(handler))
    info = remote.chat_template_info()
    assert info.template is None
    assert "does not expose /v1/chat_template" in info.note
    assert info.is_base_model is False


# --------------------------------------------------------------------------- #
# Fallback template sanity
# --------------------------------------------------------------------------- #
def test_fallback_chatml_template_mentions_roles() -> None:
    assert "<|im_start|>" in ct.FALLBACK_CHATML_TEMPLATE
    assert "add_generation_prompt" in ct.FALLBACK_CHATML_TEMPLATE


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
