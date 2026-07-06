"""Chat-template route for ``dsbx serve``.

Kept in its own module (rather than inline in :mod:`dsbx.server.app`)
partly for cohesion -- everything chat-template-shaped lives together --
and partly because ``server/app.py`` sits at its grandfathered size
ceiling and must shrink, not grow. The route factory takes the pieces it
needs (the app, the model slot, and the ready-check helper) instead of
importing them from ``app`` to avoid a circular import.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from dsbx.server import schemas as S

if TYPE_CHECKING:
    from fastapi import FastAPI

    from dsbx.core.backend import Backend


def add_chat_template_route(
    app: FastAPI,
    slot: Any,
    require_ready: Callable[[Any], Backend],
) -> None:
    """Mount ``GET /v1/chat_template`` on ``app``.

    Returns the loaded backend's chat template + special-token metadata
    (see :meth:`dsbx.core.backend.Backend.chat_template_info`). The
    handler holds the slot's inference lock like every other backend
    call so a concurrent model swap can't yank the backend mid-read.
    """

    @app.get("/v1/chat_template", response_model=S.ChatTemplateResponse)
    def chat_template() -> S.ChatTemplateResponse:
        with slot.lock:
            backend = require_ready(slot)
            info = backend.chat_template_info()
        return S.ChatTemplateResponse(**info.to_dict())
