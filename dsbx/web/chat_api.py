"""FastAPI router for the chat-mode endpoints.

Currently one endpoint:

- ``GET /api/v1/chat/template`` -- the active model's chat template +
  special-token metadata, discovered by the backend that knows where the
  model came from (HF Hub repo for cloud providers, GGUF metadata for
  llamacpp-py, the transformers tokenizer for local HF, proxied
  ``/v1/chat_template`` for remote dsbx-serve hosts).

Mounted from :func:`dsbx.web.app.make_web_app` behind the same bearer
auth as every other ``/api/v1`` route. Kept as its own module (the
``logs_api`` pattern) because ``web/app.py`` sits at its grandfathered
size ceiling.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from fastapi import APIRouter, Query

from dsbx.core.chat_template import FALLBACK_CHATML_TEMPLATE
from dsbx.web.backends import BackendRegistry
from dsbx.web.deps import use_backend
from dsbx.web.schemas_chat import ChatTemplateResponse

log = logging.getLogger("dsbx.web.chat_api")


def make_chat_router(
    registry: BackendRegistry,
    require_bearer: Callable,
) -> APIRouter:
    """Build the ``/api/v1/chat`` router bound to ``registry``."""
    from fastapi import Depends

    router = APIRouter(
        prefix="/api/v1/chat",
        tags=["chat"],
        dependencies=[Depends(require_bearer)],
    )

    @router.get("/template", response_model=ChatTemplateResponse)
    def chat_template(
        backend: str = Query(..., description="backend name from /api/v1/info"),
        model: str | None = Query(None, description="model override (cloud providers)"),
    ) -> ChatTemplateResponse:
        with use_backend(registry, backend, model=model) as be:
            info = be.chat_template_info()
        return ChatTemplateResponse(
            backend=backend,
            model=model,
            fallback_template=FALLBACK_CHATML_TEMPLATE,
            **info.to_dict(),
        )

    return router
