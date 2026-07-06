<script lang="ts">
  /**
   * The block editor: ordered ChatBlock cards + typed "add" buttons +
   * the add_generation_prompt toggle. Pure structural editing -- the
   * template rendering happens in the composer above.
   */
  import ChatBlockCard from './ChatBlockCard.svelte';
  import { freshBlockId } from '$lib/chat/types';
  import type { ChatBlock, ChatBlockKind, ChatDoc, TemplateProfile } from '$lib/chat/types';

  interface Props {
    doc: ChatDoc;
    profile: TemplateProfile | null;
    backend: string;
    model: string;
    /** Backend has a real local tokenizer -> inline token highlighting. */
    tokenizeSupported: boolean;
    disabled?: boolean;
    onChange: (doc: ChatDoc) => void;
  }
  let {
    doc,
    profile,
    backend,
    model,
    tokenizeSupported,
    disabled = false,
    onChange
  }: Props = $props();

  function newBlock(kind: ChatBlockKind): ChatBlock {
    const id = freshBlockId();
    switch (kind) {
      case 'tool_defs':
        return {
          kind,
          id,
          toolsJson:
            '[\n  {"type": "function", "function": {"name": "get_weather", "description": "", "parameters": {"type": "object", "properties": {}}}}\n]'
        };
      case 'tool_call':
        return { kind, id, name: '', argumentsJson: '{}' };
      case 'tool_result':
        return { kind, id, content: '' };
      default:
        return { kind, id, content: '' };
    }
  }

  function addBlock(kind: ChatBlockKind): void {
    onChange({ ...doc, blocks: [...doc.blocks, newBlock(kind)] });
  }

  function updateBlock(i: number, block: ChatBlock): void {
    const blocks = doc.blocks.slice();
    blocks[i] = block;
    onChange({ ...doc, blocks });
  }

  function removeBlock(i: number): void {
    onChange({ ...doc, blocks: doc.blocks.filter((_, j) => j !== i) });
  }

  function moveBlock(i: number, delta: -1 | 1): void {
    const j = i + delta;
    if (j < 0 || j >= doc.blocks.length) return;
    const blocks = doc.blocks.slice();
    [blocks[i], blocks[j]] = [blocks[j], blocks[i]];
    onChange({ ...doc, blocks });
  }

  // System button greys out (with the template's own reason) when the
  // family rejects system messages -- honest gating, not silent dropping.
  let systemBlocked = $derived(profile !== null && !profile.systemSupported);

  const addButtons: { kind: ChatBlockKind; label: string; title: string }[] = [
    { kind: 'system', label: '+ system', title: 'Add a system prompt block.' },
    { kind: 'user', label: '+ user', title: 'Add a user message.' },
    { kind: 'assistant', label: '+ assistant', title: 'Author an assistant turn yourself.' },
    {
      kind: 'assistant_reasoning',
      label: '+ reasoning',
      title: 'Thinking section for the next assistant block.'
    },
    { kind: 'tool_defs', label: '+ tools', title: 'Advertise a tool catalogue to the model.' },
    { kind: 'tool_call', label: '+ tool call', title: 'Author a native tool invocation.' },
    { kind: 'tool_result', label: '+ tool result', title: 'Feed a tool response back.' }
  ];
</script>

<div class="space-y-2">
  {#if doc.blocks.length === 0}
    <p class="text-xs text-slate-500">
      No blocks yet — add a system prompt or a user message below.
    </p>
  {/if}
  <!-- Keyed by the block's stable id when it has one (index keying would
       re-mount every card below a removal); parser-produced blocks
       without ids fall back to a namespaced index key. -->
  {#each doc.blocks as block, i (block.id ?? `i:${i}`)}
    <ChatBlockCard
      {block}
      index={i}
      count={doc.blocks.length}
      {backend}
      {model}
      {tokenizeSupported}
      {disabled}
      onChange={(b) => updateBlock(i, b)}
      onRemove={() => removeBlock(i)}
      onMove={(d) => moveBlock(i, d)}
    />
  {/each}

  <div class="flex flex-wrap gap-1.5">
    {#each addButtons as btn (btn.kind)}
      {@const blocked = btn.kind === 'system' && systemBlocked}
      <button
        type="button"
        class="text-xs px-2 py-1 rounded border border-slate-700 hover:border-slate-500 disabled:opacity-40 disabled:cursor-not-allowed"
        title={blocked
          ? 'This model\u2019s template rejects system messages (it will raise \u2014 try it in Raw mode to see the error).'
          : btn.title}
        disabled={disabled || blocked}
        onclick={() => addBlock(btn.kind)}
      >{btn.label}</button>
    {/each}
  </div>

  <label class="flex items-center gap-2 text-xs text-slate-400">
    <input
      type="checkbox"
      class="accent-sky-500"
      {disabled}
      checked={doc.addGenerationPrompt}
      onchange={(e) => onChange({ ...doc, addGenerationPrompt: e.currentTarget.checked })}
    />
    add generation prompt
    <span
      class="text-slate-600"
      title="Append the assistant header after the last message so the model starts a fresh assistant turn. Turn OFF when the last block is an assistant prefill the model should continue mid-turn."
    >(cue the model to answer)</span>
  </label>
</div>
