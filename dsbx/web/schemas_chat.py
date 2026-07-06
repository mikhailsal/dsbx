"""Pydantic wire schemas for the chat-mode endpoints.

Kept separate from :mod:`dsbx.web.schemas` (which sits at its
grandfathered size ceiling) -- everything chat-mode-shaped lives here.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dsbx.web.schemas import GenerateRequest


class ChatTemplateResponse(BaseModel):
    """``GET /api/v1/chat/template`` -- a model's chat template + metadata.

    Browser-facing mirror of
    :class:`dsbx.core.chat_template.ChatTemplateInfo`:

    - ``template`` is the raw Jinja source the browser renders client-side
      (via ``@huggingface/jinja``). ``None`` when the model ships none.
    - ``source`` names the discovery path: ``hf_hub`` (cloud provider's
      mapped tokenizer repo), ``gguf`` (llamacpp metadata),
      ``transformers`` (local HF tokenizer), ``remote`` (proxied from
      dsbx-serve), or ``none``.
    - ``is_base_model`` is true only when discovery *succeeded* and found
      no template -- the "this model was not trained for chat" signal.
      When discovery merely failed (network, gated repo, old remote
      server) it stays false and ``note`` explains what went wrong.
    - ``fallback_template`` is the sandbox's generic ChatML scaffold the
      UI offers for base-model experimentation, served here so the
      frontend and any future CLI consumer share one source of truth.
    """

    backend: str
    model: str | None = None
    template: str | None = None
    source: str = "none"
    bos_token: str | None = None
    eos_token: str | None = None
    special_tokens: dict[str, str] = Field(default_factory=dict)
    note: str = ""
    is_base_model: bool = False
    fallback_template: str


class ChatMessage(BaseModel):
    """One OpenAI-shaped conversation message for the simulation path.

    ``content`` may be ``None`` on assistant messages that carry only
    ``tool_calls``; ``tool_call_id`` marks a ``role="tool"`` result
    message. The shapes are forwarded to the provider verbatim -- the
    sandbox deliberately does NOT normalize them, because seeing exactly
    what the provider accepts/rejects is part of the lesson. Pydantic's
    default is ``extra="ignore"``, which would silently STRIP any key
    not declared here (it once ate ``reasoning_content``); ``extra=
    "allow"`` keeps the verbatim promise honest for future family-
    specific fields too.
    """

    model_config = ConfigDict(extra="allow")

    role: str
    content: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    # Reasoning/thinking text attached to an assistant message (the key
    # Qwen/DeepSeek-style providers consume). Declared explicitly --
    # rather than riding ``extra`` -- so the field is documented and
    # survives ``model_dump(exclude_none=True)`` by contract.
    reasoning_content: str | None = None


class ChatGenerateRequest(GenerateRequest):
    """:class:`GenerateRequest` + the chat-simulation fields.

    ``messages`` present == "run the chat path": the route requires a
    backend whose capabilities advertise ``supports_chat_stream`` and
    forwards the structured conversation to ``/chat/completions``.
    ``messages`` absent == the historical prompt path, bit-for-bit.
    Template-capable backends must NOT send ``messages`` -- they render
    the chat template client-side into ``prompt`` (one source of truth);
    the route rejects the combination with an explanatory 400.
    """

    messages: list[ChatMessage] | None = None
    tools: list[dict[str, Any]] | None = None


__all__ = ["ChatGenerateRequest", "ChatMessage", "ChatTemplateResponse"]
