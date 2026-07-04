<script lang="ts">
  /**
   * One editable conversation block. The per-kind affordances live here:
   * plain textareas for message roles, JSON editors (with live validation)
   * for tool definitions / calls, and the reasoning block's "attaches to
   * the next assistant turn" explainer.
   */
  import type { ChatBlock } from '$lib/chat/types';

  interface Props {
    block: ChatBlock;
    index: number;
    count: number;
    disabled?: boolean;
    onChange: (block: ChatBlock) => void;
    onRemove: () => void;
    onMove: (delta: -1 | 1) => void;
  }
  let { block, index, count, disabled = false, onChange, onRemove, onMove }: Props = $props();

  const meta: Record<ChatBlock['kind'], { label: string; accent: string; hint: string }> = {
    system: {
      label: 'system',
      accent: 'border-l-violet-500/60',
      hint: 'Instructions injected before the conversation.'
    },
    user: { label: 'user', accent: 'border-l-sky-500/60', hint: 'A human turn.' },
    assistant: {
      label: 'assistant',
      accent: 'border-l-emerald-500/60',
      hint: 'An AI turn — write it yourself to put words in the model\u2019s mouth.'
    },
    assistant_reasoning: {
      label: 'reasoning',
      accent: 'border-l-fuchsia-500/60',
      hint: 'Thinking section attached to the NEXT assistant block (family convention, e.g. <think>…</think>).'
    },
    tool_defs: {
      label: 'tools',
      accent: 'border-l-amber-500/60',
      hint: 'JSON array of function definitions advertised to the model.'
    },
    tool_call: {
      label: 'tool call',
      accent: 'border-l-orange-500/60',
      hint: 'A native tool invocation authored on behalf of the assistant.'
    },
    tool_result: {
      label: 'tool result',
      accent: 'border-l-teal-500/60',
      hint: 'The tool\u2019s reply, fed back to the model.'
    }
  };

  function jsonError(text: string, wantArray: boolean): string {
    try {
      const v: unknown = JSON.parse(text);
      if (wantArray && !Array.isArray(v)) return 'expected a JSON array';
      return '';
    } catch (e) {
      return e instanceof Error ? e.message : String(e);
    }
  }
</script>

<div class="rounded border border-slate-800 border-l-2 {meta[block.kind].accent} bg-slate-900/40 p-2 space-y-1.5">
  <div class="flex items-center gap-2">
    <span class="text-[10px] uppercase tracking-wider text-slate-400" title={meta[block.kind].hint}>
      {meta[block.kind].label}
    </span>
    <span class="flex-1"></span>
    <button
      type="button"
      class="text-xs px-1 text-slate-500 hover:text-slate-200 disabled:opacity-30"
      title="move up"
      aria-label="move block up"
      onclick={() => onMove(-1)}
      disabled={disabled || index === 0}
    >↑</button>
    <button
      type="button"
      class="text-xs px-1 text-slate-500 hover:text-slate-200 disabled:opacity-30"
      title="move down"
      aria-label="move block down"
      onclick={() => onMove(1)}
      disabled={disabled || index === count - 1}
    >↓</button>
    <button
      type="button"
      class="text-xs px-1 text-slate-500 hover:text-rose-300"
      title="delete block"
      aria-label="delete block"
      onclick={onRemove}
      disabled={disabled}
    >×</button>
  </div>

  {#if block.kind === 'tool_defs'}
    {@const err = jsonError(block.toolsJson, true)}
    <textarea
      class="input w-full font-mono text-xs"
      rows="4"
      spellcheck="false"
      {disabled}
      value={block.toolsJson}
      oninput={(e) => onChange({ ...block, toolsJson: e.currentTarget.value })}
    ></textarea>
    {#if err}
      <p class="text-[11px] text-rose-400">invalid JSON: {err}</p>
    {/if}
  {:else if block.kind === 'tool_call'}
    {@const err = jsonError(block.argumentsJson, false)}
    <div class="grid grid-cols-[1fr_2fr] gap-1.5">
      <input
        type="text"
        class="input font-mono text-xs"
        placeholder="function name"
        {disabled}
        value={block.name}
        oninput={(e) => onChange({ ...block, name: e.currentTarget.value })}
      />
      <input
        type="text"
        class="input font-mono text-xs"
        placeholder={'arguments JSON, e.g. {"city": "Paris"}'}
        {disabled}
        value={block.argumentsJson}
        oninput={(e) => onChange({ ...block, argumentsJson: e.currentTarget.value })}
      />
    </div>
    {#if err}
      <p class="text-[11px] text-rose-400">invalid arguments JSON: {err}</p>
    {/if}
  {:else if block.kind === 'tool_result'}
    <textarea
      class="input w-full font-mono text-xs"
      rows="2"
      placeholder="tool output"
      {disabled}
      value={block.content}
      oninput={(e) => onChange({ ...block, content: e.currentTarget.value })}
    ></textarea>
    <input
      type="text"
      class="input w-full font-mono text-[11px]"
      placeholder="tool name (optional)"
      {disabled}
      value={block.name ?? ''}
      oninput={(e) => onChange({ ...block, name: e.currentTarget.value || undefined })}
    />
  {:else}
    <textarea
      class="input w-full text-sm"
      rows={block.kind === 'assistant_reasoning' ? 2 : 3}
      placeholder={meta[block.kind].hint}
      {disabled}
      value={block.content}
      oninput={(e) => onChange({ ...block, content: e.currentTarget.value })}
    ></textarea>
  {/if}
</div>
