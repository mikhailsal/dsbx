"""Small shared route helpers for the web middleware.

Extracted from ``web/app.py`` so satellite routers (``chat_api``,
future feature routers) can reuse the same backend-acquisition contract
without importing the (large) app factory module and without duplicating
the error mapping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import HTTPException

if TYPE_CHECKING:
    from dsbx.web.backends import BackendRegistry, _LockedBackend


def use_backend(registry: BackendRegistry, name: str, model: str | None = None) -> _LockedBackend:
    """``with use_backend(...) as backend:`` -- lock + load.

    ``model`` is honored only for cloud providers (see
    :meth:`BackendRegistry.use`); other families ignore it but won't error,
    so callers can pass the request's ``model`` field through without
    branching on family. Unknown backends map to 404, family/kind misuse
    to 400 -- the same contract every ``/api/v1`` route relies on.
    """
    try:
        return registry.use(name, model=model)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
