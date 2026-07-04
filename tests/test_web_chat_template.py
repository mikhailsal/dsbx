"""Tests for the web middleware's ``GET /api/v1/chat/template`` endpoint."""

from __future__ import annotations

from dsbx.core.chat_template import FALLBACK_CHATML_TEMPLATE, ChatTemplateInfo
from tests.fakes import FakeBackend
from tests.web_helpers import build_test_app, make_authed_client

CHATML = "{% for message in messages %}...{% endfor %}"


def _app_with(chat_template: ChatTemplateInfo | None):
    fake = FakeBackend(chat_template=chat_template)
    app = build_test_app({"dsbx-host-py": fake})
    return make_authed_client(app)


def test_requires_auth() -> None:
    app = build_test_app({"dsbx-host-py": FakeBackend()})
    from fastapi.testclient import TestClient

    r = TestClient(app).get("/api/v1/chat/template", params={"backend": "dsbx-host-py"})
    assert r.status_code == 401


def test_returns_template_and_fallback() -> None:
    client = _app_with(
        ChatTemplateInfo(
            template=CHATML,
            source="gguf",
            bos_token="<|im_start|>",
            eos_token="<|im_end|>",
            special_tokens={"eos_token": "<|im_end|>"},
        )
    )
    r = client.get("/api/v1/chat/template", params={"backend": "dsbx-host-py"})
    assert r.status_code == 200
    data = r.json()
    assert data["backend"] == "dsbx-host-py"
    assert data["template"] == CHATML
    assert data["source"] == "gguf"
    assert data["bos_token"] == "<|im_start|>"
    assert data["is_base_model"] is False
    # The generic ChatML scaffold ships on every response so the UI's
    # base-model fallback never needs a second endpoint.
    assert data["fallback_template"] == FALLBACK_CHATML_TEMPLATE


def test_base_model_flag_set_when_no_template() -> None:
    client = _app_with(None)  # FakeBackend defaults to none_info()
    r = client.get("/api/v1/chat/template", params={"backend": "dsbx-host-py"})
    assert r.status_code == 200
    data = r.json()
    assert data["template"] is None
    assert data["is_base_model"] is True
    assert data["fallback_template"]


def test_degraded_discovery_is_not_base_model() -> None:
    client = _app_with(ChatTemplateInfo(template=None, source="none", note="hub unreachable"))
    r = client.get("/api/v1/chat/template", params={"backend": "dsbx-host-py"})
    data = r.json()
    assert data["is_base_model"] is False
    assert data["note"] == "hub unreachable"


def test_unknown_backend_404() -> None:
    client = _app_with(None)
    r = client.get("/api/v1/chat/template", params={"backend": "nope"})
    assert r.status_code == 404


def test_model_param_forwarded_to_cloud_variant() -> None:
    """The ``model`` query param must reach the registry's variant lookup."""
    client = _app_with(ChatTemplateInfo(template=CHATML, source="hf_hub"))
    r = client.get(
        "/api/v1/chat/template",
        params={"backend": "dsbx-host-py", "model": "some/model"},
    )
    # Remote-family backends ignore the model override (no error), and the
    # response echoes it back for cache-keying on the client side.
    assert r.status_code == 200
    assert r.json()["model"] == "some/model"
