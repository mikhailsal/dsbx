"""Chat-template discovery shared by every backend.

A *chat template* is the Jinja program shipped with a model's tokenizer that
turns a structured conversation (system / user / assistant / tool messages)
into the single raw token stream the model actually consumes. The Decode
workbench's chat mode renders templates client-side (via ``@huggingface/jinja``)
but needs the template *source* + special-token metadata from the backend
that knows where the model came from:

- cloud OpenAI-compat backends fetch ``tokenizer_config.json`` (and, for
  newer repos, the standalone ``chat_template.jinja``) from the mapped
  HuggingFace repo;
- the local HF backend reads ``tokenizer.chat_template``;
- llamacpp-py reads the ``tokenizer.chat_template`` GGUF metadata key;
- the remote backend proxies ``GET /v1/chat_template`` from dsbx-serve.

``template is None`` doubles as the *base-model detector*: a repo that
ships no chat template was (almost certainly) never trained on chat
formatting, so the UI shows a warning and offers the explicit ChatML
fallback below for experimentation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# A deliberately minimal ChatML template used when the model ships none.
# Explicitly labeled in the UI as a fallback: base models were not trained
# on ANY chat format, so this is a pedagogical "what if" scaffold, not a
# claim about the model's training data.
FALLBACK_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

# tokenizer_config.json keys whose values name single special tokens worth
# surfacing to the chat UI (snippet buttons / stop hints).
_SPECIAL_TOKEN_KEYS = ("bos_token", "eos_token", "unk_token", "pad_token", "sep_token")


@dataclass
class ChatTemplateInfo:
    """Everything the chat UI needs to know about a model's template.

    ``source`` says where the template came from (``hf_hub`` / ``gguf`` /
    ``transformers`` / ``remote`` / ``none``); ``note`` carries a
    human-readable explanation when discovery degraded (network failure,
    gated repo, old remote server) so the UI can tell the user *why*
    there's no template instead of silently claiming "base model".
    """

    template: str | None
    source: str
    bos_token: str | None = None
    eos_token: str | None = None
    special_tokens: dict[str, str] = field(default_factory=dict)
    note: str = ""

    @property
    def is_base_model(self) -> bool:
        """No template found (and discovery didn't merely fail)."""
        return self.template is None and not self.note

    def to_dict(self) -> dict[str, Any]:
        return {
            "template": self.template,
            "source": self.source,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "special_tokens": dict(self.special_tokens),
            "note": self.note,
            "is_base_model": self.is_base_model,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChatTemplateInfo:
        return cls(
            template=d.get("template"),
            source=str(d.get("source", "none")),
            bos_token=d.get("bos_token"),
            eos_token=d.get("eos_token"),
            special_tokens={str(k): str(v) for k, v in (d.get("special_tokens") or {}).items()},
            note=str(d.get("note", "")),
        )


def none_info(note: str = "") -> ChatTemplateInfo:
    """The "no template available" shape every degradation path returns."""
    return ChatTemplateInfo(template=None, source="none", note=note)


def token_text(value: Any) -> str | None:
    """Normalize a tokenizer_config token value to its surface string.

    HF configs store special tokens either as a plain string or as an
    AddedToken-style object ``{"content": "...", ...}``. Anything else
    (None, unexpected shapes) maps to ``None``.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        return str(content) if isinstance(content, str) else None
    return None


def template_from_config_value(value: Any) -> str | None:
    """Extract the template string from a ``chat_template`` config value.

    The key holds a plain Jinja string, a list of ``{"name": ...,
    "template": ...}`` entries (multi-template repos), or -- via older
    transformers tokenizers -- a ``{name: template}`` dict. For the
    multi-template shapes we prefer the entry named ``default`` and
    otherwise take the first.
    """
    if isinstance(value, str):
        return value or None
    named: dict[str, Any] = {}
    if isinstance(value, list):
        named = {
            str(entry.get("name", "")): entry.get("template")
            for entry in value
            if isinstance(entry, dict)
        }
    elif isinstance(value, dict):
        named = {str(k): v for k, v in value.items()}
    chosen = named.get("default") or next(iter(named.values()), None)
    return str(chosen) if isinstance(chosen, str) and chosen else None


def info_from_tokenizer_config(
    config: dict[str, Any],
    *,
    source: str,
    template_override: str | None = None,
) -> ChatTemplateInfo:
    """Build a :class:`ChatTemplateInfo` from a tokenizer_config.json dict.

    ``template_override`` wins over the config's own ``chat_template``
    key -- used when the repo ships the template as a standalone
    ``chat_template.jinja`` file (the newer transformers convention).
    """
    template = template_override or template_from_config_value(config.get("chat_template"))
    specials: dict[str, str] = {}
    for key in _SPECIAL_TOKEN_KEYS:
        text = token_text(config.get(key))
        if text:
            specials[key] = text
    return ChatTemplateInfo(
        template=template,
        source=source if template else "none",
        bos_token=specials.get("bos_token"),
        eos_token=specials.get("eos_token"),
        special_tokens=specials,
    )


def fetch_hf_chat_template(repo_id: str) -> ChatTemplateInfo:
    """Fetch chat-template metadata for ``repo_id`` from the HuggingFace Hub.

    Looks at ``tokenizer_config.json`` first (the classic home of the
    ``chat_template`` key + special-token names), then falls back to the
    standalone ``chat_template.jinja`` file newer repos ship instead.
    Network / gating failures degrade to :func:`none_info` with the error
    recorded in ``note`` -- callers cache the result so the Hub is hit at
    most once per backend instance.
    """
    from huggingface_hub import hf_hub_download

    config: dict[str, Any] = {}
    try:
        path = hf_hub_download(repo_id=repo_id, filename="tokenizer_config.json")
        config = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("chat-template fetch failed for %s: %s", repo_id, exc)
        return none_info(note=f"could not fetch tokenizer_config.json ({type(exc).__name__})")

    template_override: str | None = None
    if not config.get("chat_template"):
        template_override = _fetch_standalone_template(repo_id)
    return info_from_tokenizer_config(config, source="hf_hub", template_override=template_override)


def _fetch_standalone_template(repo_id: str) -> str | None:
    """Best-effort fetch of the standalone ``chat_template.jinja`` file."""
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(repo_id=repo_id, filename="chat_template.jinja")
    except Exception:
        # Missing file is the normal case for older repos; genuine network
        # failures were already surfaced by the tokenizer_config fetch.
        return None
    text = Path(path).read_text(encoding="utf-8")
    return text or None


__all__ = [
    "FALLBACK_CHATML_TEMPLATE",
    "ChatTemplateInfo",
    "fetch_hf_chat_template",
    "info_from_tokenizer_config",
    "none_info",
    "template_from_config_value",
    "token_text",
]
