# Chat mode: templates are just text

The Decode tab's `Text | Chat` toggle adds a white-box chat workbench on
top of the existing decoding pipeline. The pedagogical point it exists to
make: **a "chat" with an LLM is nothing but a text-continuation request
whose prompt was assembled by a template.** Change the template and you
change the model's behaviour -- same weights, same sampler, different
text in.

## How it works

```
blocks (system / user / assistant / reasoning / tools)
        │  render: @huggingface/jinja, IN THE BROWSER,
        │  through the model's REAL chat template
        ▼
raw prompt string  ──────────────►  POST /api/v1/generate/stream
        ▲                            (the same `prompt` field text
        │  parse: derived markers     mode uses -- manual picking,
raw editor (freely editable)          watch, bias, stops all work)
```

- **Template discovery** (`GET /api/v1/chat/template`): the middleware
  fetches the Jinja `chat_template` + special tokens from the HF Hub
  `tokenizer_config.json` (cloud providers, via the
  `[providers.*.tokenizers]` mapping), GGUF metadata (`llamacpp-py`),
  the `transformers` tokenizer (local HF), or a remote `dsbx serve`
  host's `/v1/chat_template`. The template panel in the UI shows the
  source so you always know where the truth came from.
- **Rendering happens client-side** with `@huggingface/jinja` (the same
  engine transformers.js uses). No server round-trip per keystroke, and
  what you see in the Raw editor is byte-for-byte what will be sent.
- **Marker derivation** (`frontend/src/lib/chat/profile.ts`): rather
  than hard-coding per-model formats, dsbx renders sentinel
  conversations through the real template and diffs them to recover
  each role's prefix/suffix, the generation prompt, and system-message
  behaviour. Markers are taken from *mid-conversation* turns because
  several templates (Qwen3/3.5) render the last assistant turn
  specially (empty `<think>` scaffolds). A curated per-family table
  adds what templates can't say mechanically: how reasoning
  (`<think>`, Harmony `analysis` channels) and tool calls
  (`<tool_call>` JSON, `[TOOL_CALLS]`) are embedded in assistant
  content.
- **Round-trip self-test**: a derived profile is only trusted
  (`complete: true`) after render -> parse -> compare succeeds on a
  sentinel conversation. Anything else honestly degrades to raw-only
  editing with a note.
- **Blocks ⇄ Raw**: switching to Raw renders the blocks; switching back
  parses the text with the derived markers. Parse failures produce a
  positioned error (offset, expected, found, hint) and never discard
  your text; unknown constructs degrade to plain assistant text with a
  warning.
- **Snippet buttons** in Raw mode are generated from the derived
  profile, so "User turn" / "Cue model" / "Thinking" / "Tool call"
  insert markers that are correct for the *loaded model*, by
  construction.

## Raw mode vs simulation mode

| Backend | Chat mode behaviour |
|---|---|
| Fireworks, local HF, `llamacpp-py`, remote `dsbx serve` | Full raw mode: the rendered string rides the existing `prompt` field, so **every** decode feature works -- inspect (prompt logits), manual token picking mid-assistant-turn, watch columns, logit bias, stop tokens. The Raw editor is fully editable: corrupt the template on purpose and watch what happens. |
| NVIDIA NIM, OpenRouter | **Simulation mode**: these providers only accept `/chat/completions` and render the template server-side (OpenRouter's upstreams silently rewrite raw prompts into chat messages). dsbx sends your blocks as structured `messages[]`, streams real per-token `top_logprobs` into the same step tables, and shows a read-only Raw preview labeled as *the most likely form* of what the provider renders. Features that need a client-visible token stream (manual picking, prepend ids) are hidden with explanatory notes. |

Sampler support in simulation mode is limited to samplers with a native
`/chat/completions` analogue (greedy, temperature, top_p); anything else
would silently degrade server-side, so the run buttons gate on it
honestly.

## Base models

A model with no chat template (e.g. a `*-Base` GGUF) gets a warning
banner -- it was never trained on chat markers, so a generic ChatML
fallback template is offered purely for experimentation. Watching a base
model try (and fail) to respect `<|im_start|>` fences is itself a good
lesson in what instruction tuning actually adds.

## Where the code lives

| Piece | Location |
|---|---|
| Template discovery (Python) | `dsbx/core/chat_template.py`, `Backend.chat_template_info()`, `dsbx/web/chat_api.py` |
| Chat-completions streaming (NIM/OpenRouter) | `dsbx/backends/openai_compat/_streaming_chat.py`, `dsbx/web/streaming_chat.py` |
| Client-side engine | `frontend/src/lib/chat/` (`render.ts`, `profile.ts`, `parse.ts`, `snippets.ts`) |
| UI components | `frontend/src/lib/components/chat/` (`ChatComposer`, `ChatBlockList`, `ChatBlockCard`, `ChatRawEditor`, `ChatTemplatePanel`, `BaseModelBanner`) |
| Shared request/stream plumbing | `frontend/src/lib/generate/` (`request.ts`, `stream.ts`, `watch.ts`) |
