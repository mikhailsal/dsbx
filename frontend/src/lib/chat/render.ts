/**
 * Blocks -> messages -> raw string, through the model's own chat template.
 *
 * This is the forward half of the engine: the block editor's ``ChatDoc``
 * becomes (1) an OpenAI-shaped ``messages[]`` list -- used directly by the
 * simulation path for chat-only providers -- and (2) the raw prompt string
 * produced by executing the model's Jinja template in the browser via
 * ``@huggingface/jinja`` (the engine transformers.js uses). Template-capable
 * backends receive that raw string on the existing ``prompt`` field, which is
 * exactly the point pedagogically: chat is just text, and here is the text.
 */

import { Template } from '@huggingface/jinja';
import type {
  ChatBlock,
  ChatDoc,
  ChatMessage,
  RenderResult,
  TemplateInputs,
  ToolCallPayload
} from './types';

/**
 * Generic ChatML scaffold offered when the model ships no template.
 * Explicitly labeled in the UI as a fallback: a base model was not trained
 * on ANY chat format, so this is a "what if" scaffold for experimentation,
 * not a claim about the model. Mirrors ``FALLBACK_CHATML_TEMPLATE`` in
 * ``dsbx/core/chat_template.py`` (the backend serves it too, so the two
 * stay in sync via the template endpoint's ``fallback_template`` field).
 */
export const FALLBACK_CHATML_TEMPLATE =
  "{% for message in messages %}" +
  "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}" +
  "{% endfor %}" +
  "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

/**
 * Fold the block list into OpenAI-shaped messages.
 *
 * Adjacency rules (the interesting part):
 * - ``assistant_reasoning`` attaches to the NEXT ``assistant`` block as its
 *   ``reasoning_content`` (the key Qwen3/DeepSeek templates consume); a
 *   trailing reasoning block becomes an assistant message of its own.
 * - consecutive ``tool_call`` blocks merge into ONE assistant message with a
 *   ``tool_calls`` array (matching how models emit parallel calls).
 * - ``tool_defs`` contributes to ``tools``, not to ``messages``.
 */
export function blocksToMessages(blocks: ChatBlock[]): {
  messages: ChatMessage[];
  tools: Record<string, unknown>[] | null;
  warnings: string[];
} {
  const messages: ChatMessage[] = [];
  const warnings: string[] = [];
  let tools: Record<string, unknown>[] | null = null;
  let pendingReasoning: string | null = null;
  let callSeq = 0;

  const flushReasoning = (): void => {
    if (pendingReasoning !== null) {
      warnings.push(
        'reasoning block is not followed by an assistant block; emitted as a reasoning-only assistant message.'
      );
      messages.push({ role: 'assistant', content: '', reasoning_content: pendingReasoning });
      pendingReasoning = null;
    }
  };

  for (const block of blocks) {
    switch (block.kind) {
      case 'system':
      case 'user':
        flushReasoning();
        messages.push({ role: block.kind, content: block.content });
        break;
      case 'assistant': {
        const msg: ChatMessage = { role: 'assistant', content: block.content };
        if (pendingReasoning !== null) {
          msg.reasoning_content = pendingReasoning;
          pendingReasoning = null;
        }
        messages.push(msg);
        break;
      }
      case 'assistant_reasoning':
        flushReasoning();
        pendingReasoning = block.content;
        break;
      case 'tool_defs': {
        const parsed = parseJsonArray(block.toolsJson);
        if (parsed.error) {
          warnings.push(`tool definitions ignored: ${parsed.error}`);
        } else {
          tools = [...(tools ?? []), ...parsed.value];
        }
        break;
      }
      case 'tool_call': {
        flushReasoning();
        const call: ToolCallPayload = {
          id: block.callId || `call_${++callSeq}`,
          type: 'function',
          function: { name: block.name, arguments: block.argumentsJson }
        };
        const prev = messages[messages.length - 1];
        if (prev && prev.role === 'assistant' && Array.isArray(prev.tool_calls)) {
          prev.tool_calls.push(call);
        } else {
          messages.push({ role: 'assistant', content: null, tool_calls: [call] });
        }
        break;
      }
      case 'tool_result': {
        flushReasoning();
        const msg: ChatMessage = { role: 'tool', content: block.content };
        if (block.name) msg.name = block.name;
        if (block.callId) msg.tool_call_id = block.callId;
        messages.push(msg);
        break;
      }
    }
  }
  flushReasoning();
  return { messages, tools, warnings };
}

function parseJsonArray(text: string): {
  value: Record<string, unknown>[];
  error: string | null;
} {
  try {
    const parsed: unknown = JSON.parse(text);
    if (!Array.isArray(parsed)) {
      return { value: [], error: 'expected a JSON array of tool definitions' };
    }
    return { value: parsed as Record<string, unknown>[], error: null };
  } catch (e) {
    return { value: [], error: e instanceof Error ? e.message : String(e) };
  }
}

/**
 * Render a ``ChatDoc`` through the model's template (or the ChatML
 * fallback when the model ships none). Never throws: template execution
 * failures -- including deliberate ``raise_exception`` calls like Gemma's
 * "System role not supported" -- land in ``error`` with ``raw = ''`` so
 * the UI can show the template's own complaint verbatim.
 *
 * ``@huggingface/jinja`` ships the HF-specific globals itself
 * (``raise_exception``, ``strftime_now`` for gpt-oss date stamping), so
 * the context only carries the conversation and the tokenizer strings.
 */
export function renderChat(
  doc: ChatDoc,
  inputs: TemplateInputs,
  assistantPrefix: string | null = null
): RenderResult {
  const { messages, tools, warnings } = blocksToMessages(doc.blocks);
  const usedFallback = !inputs.template;
  if (usedFallback) {
    warnings.push('model ships no chat template; using the generic ChatML fallback.');
  }
  const source = inputs.template ?? FALLBACK_CHATML_TEMPLATE;
  const prefill = prefillContent(doc, warnings);
  // When the profile knows the assistant turn opener, an open turn is
  // rendered as ``prefix + content`` -- so any scaffold the generation
  // prompt would inject (Qwen's "<think>\n") lives IN the visible block
  // content instead of being conjured invisibly. Without a derivable
  // prefix, fall back to ``generation prompt + content``.
  const usePrefix = prefill !== null && !!assistantPrefix;

  let raw = '';
  let error: string | null = null;
  try {
    const template = new Template(source);
    const context: Record<string, unknown> = {
      // A trailing prefill block is NOT rendered through the template:
      // templates close (or mangle -- Qwen strips non-final <think>
      // sections) every message they render. Instead the prior turns
      // render normally and the prefill text is appended verbatim --
      // reproducing the exact token context of the mid-turn model,
      // open reasoning sections included.
      messages: prefill === null ? messages : messages.slice(0, -1),
      add_generation_prompt: prefill === null ? doc.addGenerationPrompt : !usePrefix,
      bos_token: inputs.bosToken ?? '',
      eos_token: inputs.eosToken ?? ''
    };
    if (tools) context.tools = tools;
    raw = template.render(context);
    if (prefill !== null) raw += usePrefix ? assistantPrefix + prefill : prefill;
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }
  return { raw, messages, tools, usedFallback, warnings, error };
}

/**
 * The trailing prefill text, or ``null`` when the doc ends in a closed
 * turn. Prefill flags anywhere else are ignored with a warning -- an open
 * turn mid-conversation is not representable in template output.
 */
function prefillContent(doc: ChatDoc, warnings: string[]): string | null {
  const blocks = doc.blocks;
  const openAt = blocks.findIndex((b) => b.kind === 'assistant' && b.prefill);
  if (openAt === -1) return null;
  const last = blocks[blocks.length - 1];
  if (openAt !== blocks.length - 1 || last.kind !== 'assistant' || !last.prefill) {
    warnings.push(
      'prefill flag ignored: only the FINAL assistant block can be an open (prefill) turn.'
    );
    return null;
  }
  if (doc.addGenerationPrompt) {
    warnings.push(
      'prefill assistant block supersedes "add generation prompt" (the open turn already cues the model).'
    );
  }
  return last.content;
}
