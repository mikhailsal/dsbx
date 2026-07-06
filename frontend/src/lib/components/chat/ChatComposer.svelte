<script lang="ts">
  /**
   * Chat-mode composer for the Decode tab -- the orchestrator that owns
   * the conversation state and produces the request payload:
   *
   * - template-capable backends: the conversation renders through the
   *   model's own chat template (client-side Jinja) into ``prompt`` --
   *   the EXISTING wire field, so every decode feature (manual picking,
   *   prompt logits, watch, bias) works on it unchanged;
   * - simulation backends (NIM / OpenRouter): the structured
   *   ``messages[]`` are the payload, and the raw view is a read-only
   *   "most likely form" reconstruction, because the provider renders
   *   the real template server-side.
   *
   * Sub-modes: Blocks (structured cards) <-> Raw (direct text editing with
   * model-correct snippet buttons). Switching Raw -> Blocks parses the
   * text and refuses to switch on error, showing the position + hint
   * instead of destroying the user's edit.
   */
  import { untrack } from 'svelte';
  import BaseModelBanner from './BaseModelBanner.svelte';
  import ChatBlockList from './ChatBlockList.svelte';
  import ChatRawEditor from './ChatRawEditor.svelte';
  import ChatTemplatePanel from './ChatTemplatePanel.svelte';
  import { fetchChatTemplate, templateInputsOf } from '$lib/chat/template';
  import { deriveProfile, prefillMarkers } from '$lib/chat/profile';
  import { appendAssistantText } from '$lib/chat/append';
  import { renderChat } from '$lib/chat/render';
  import { parseRaw } from '$lib/chat/parse';
  import { buildSnippets } from '$lib/chat/snippets';
  import { withBlockIds } from '$lib/chat/types';
  import type { ChatDoc, ChatMessage, ParseError } from '$lib/chat/types';
  import type { ChatTemplateResponse } from '$lib/types';

  interface Props {
    backend: string;
    model: string;
    /** Chat-only provider: messages[] payload + read-only raw preview. */
    simulation: boolean;
    /** Backend has a real local tokenizer -> token-boundary highlighting. */
    tokenizeSupported: boolean;
    /** The backend's currently-loaded model (remote / local backends can
     * swap it while ``model`` stays constant); used purely as a template
     * cache-freshness key so a swap triggers a refetch. */
    loadedModel?: string | null;
    disabled?: boolean;
    /** Rendered raw prompt (bindable output; template-capable path). */
    prompt: string;
    /** Structured conversation (bindable output; simulation path). */
    messages: ChatMessage[] | null;
    tools: Record<string, unknown>[] | null;
    /** Whether the composer's current state is runnable. */
    ready: boolean;
  }
  let {
    backend,
    model,
    simulation,
    tokenizeSupported,
    loadedModel = null,
    disabled = false,
    prompt = $bindable(),
    messages = $bindable(),
    tools = $bindable(),
    ready = $bindable()
  }: Props = $props();

  let info = $state<ChatTemplateResponse | null>(null);
  let fetchError = $state('');
  let subMode = $state<'blocks' | 'raw'>('blocks');
  let rawText = $state('');
  let parseError = $state<ParseError | null>(null);
  let parseWarnings = $state<string[]>([]);
  let doc = $state<ChatDoc>(
    withBlockIds({
      blocks: [{ kind: 'user', content: 'What is the capital of France?' }],
      addGenerationPrompt: true
    })
  );

  // ---- template discovery (cached per backend+model+loaded model) ------ //
  $effect(() => {
    const b = backend;
    const m = model;
    const lm = loadedModel ?? '';
    if (!b) return;
    info = null;
    fetchError = '';
    fetchChatTemplate(b, m || null, lm).then(
      (res) => {
        // Ignore stale responses after a backend/model switch.
        if (backend === b && model === m && (loadedModel ?? '') === lm) info = res;
      },
      (e: unknown) => {
        if (backend === b && model === m && (loadedModel ?? '') === lm) {
          fetchError = e instanceof Error ? e.message : String(e);
        }
      }
    );
  });

  let inputs = $derived(info ? templateInputsOf(info) : null);
  let profile = $derived(inputs ? deriveProfile(inputs) : null);
  let snippets = $derived(profile && inputs ? buildSnippets(profile, inputs) : []);
  // Open-turn strings (assistant prefix + generation-prompt scaffold);
  // null degrades prefill rendering to "generation prompt + content".
  let openTurn = $derived(prefillMarkers(profile));
  let rendered = $derived(inputs ? renderChat(doc, inputs, openTurn?.prefix ?? null) : null);

  // Raw editing is only offered when the profile round-trips reliably;
  // otherwise (weird template) blocks stay the single source of truth
  // and the raw tab becomes a read-only preview like simulation mode.
  let rawEditable = $derived(!simulation && (profile?.complete ?? false));

  // ---- output contract (the page's buildRequest reads these) ----------- //
  $effect(() => {
    if (simulation) {
      prompt = '';
      messages = rendered?.messages ?? null;
      tools = rendered?.tools ?? null;
      ready = (rendered?.messages.length ?? 0) > 0;
    } else if (subMode === 'raw' && rawEditable) {
      prompt = rawText;
      messages = null;
      tools = null;
      ready = rawText.length > 0;
    } else {
      prompt = rendered?.raw ?? '';
      messages = null;
      tools = null;
      ready = !!rendered && !rendered.error && rendered.raw.length > 0;
    }
  });

  // ---- sub-mode switching ---------------------------------------------- //
  function switchToRaw(): void {
    rawText = rendered?.raw ?? '';
    parseError = null;
    parseWarnings = [];
    subMode = 'raw';
  }

  function switchToBlocks(): void {
    if (!rawEditable || !profile) {
      subMode = 'blocks';
      return;
    }
    const result = parseRaw(rawText, profile);
    if (!result.ok) {
      parseError = result.error;
      return; // stay in raw; the editor shows the positioned error + escape
    }
    doc = withBlockIds(result.doc);
    parseError = null;
    parseWarnings = result.warnings;
    subMode = 'blocks';
  }

  /** Escape hatch from an unparseable raw edit: drop the raw text and
   * return to the blocks that produced it (they are still intact). */
  function discardRawEdits(): void {
    rawText = rendered?.raw ?? '';
    parseError = null;
    parseWarnings = [];
    subMode = 'blocks';
  }

  /** Nuke the whole conversation back to a fresh user turn. */
  function resetConversation(): void {
    if (!window.confirm('Reset the conversation? All blocks and raw edits are discarded.')) return;
    doc = withBlockIds({ blocks: [{ kind: 'user', content: '' }], addGenerationPrompt: true });
    rawText = '';
    parseError = null;
    parseWarnings = [];
    subMode = 'blocks';
  }

  /** "append as assistant block" -- the chat-mode sibling of the text
   * mode's "move to prompt" (exposed on the component instance).
   *
   * ``finished`` says whether the source run ended the turn naturally
   * (EOS / stop token). An UNFINISHED run (max_tokens, cancelled)
   * becomes an open prefill block instead of a closed turn -- and a
   * follow-up run's text MERGES into that same open turn instead of
   * splitting the assistant message (see lib/chat/append.ts). */
  export function appendAssistant(text: string, finished: boolean): void {
    if (!text) return;
    if (subMode === 'raw' && rawEditable) {
      rawText = rawText + text;
      return;
    }
    doc = withBlockIds(appendAssistantText(doc, text, finished, profile, inputs?.eosToken ?? null));
  }

  // Reset conversation state when the backend/model changes enough that
  // the old raw text would be in the WRONG template language. Blocks
  // survive the switch (they're template-independent); raw text is
  // re-rendered from blocks via the new template.
  let templateKey = $derived(inputs?.template ?? '');
  let lastTemplateKey = '';
  $effect(() => {
    const key = templateKey;
    untrack(() => {
      if (key !== lastTemplateKey && subMode === 'raw') {
        rawText = rendered?.raw ?? '';
        parseError = null;
      }
      lastTemplateKey = key;
    });
  });
</script>

<div class="space-y-2">
  {#if fetchError}
    <div class="rounded border border-rose-700/50 bg-rose-950/30 px-2.5 py-2 text-xs text-rose-300">
      chat template fetch failed: {fetchError}
    </div>
  {:else if !info}
    <div class="text-xs text-slate-500 animate-pulse">discovering chat template…</div>
  {:else}
    <BaseModelBanner {info} />

    {#if simulation}
      <div class="rounded border border-sky-700/40 bg-sky-950/30 px-3 py-2 text-xs text-sky-300 leading-snug">
        <span class="font-semibold uppercase tracking-wider text-[10px]">simulation mode</span>
        — this provider is chat-only: it renders the real template
        server-side and we send the structured messages. The Raw tab shows
        the <em>most likely</em> raw form (reconstructed locally from the
        mapped tokenizer's template) and is read-only.
      </div>
    {/if}

    <div class="flex items-center gap-1">
      <button
        type="button"
        class="text-xs px-2.5 py-1 rounded-l border border-slate-700 {subMode === 'blocks'
          ? 'bg-slate-700/60 text-slate-100'
          : 'text-slate-400 hover:text-slate-200'}"
        onclick={switchToBlocks}
        {disabled}
      >Blocks</button>
      <button
        type="button"
        class="text-xs px-2.5 py-1 rounded-r border border-slate-700 -ml-px {subMode === 'raw'
          ? 'bg-slate-700/60 text-slate-100'
          : 'text-slate-400 hover:text-slate-200'}"
        onclick={switchToRaw}
        {disabled}
        title={rawEditable
          ? 'Edit the rendered template text directly.'
          : 'Read-only preview (simulation mode or a template the parser cannot round-trip).'}
      >Raw{rawEditable ? '' : ' (read-only)'}</button>
      {#if profile && !profile.complete && !simulation}
        <span
          class="ml-2 text-[10px] text-amber-400"
          title={profile.notes.join(' ')}
        >template markers not fully derivable — blocks are the source of truth</span>
      {/if}
      <button
        type="button"
        class="ml-auto text-[11px] px-2 py-0.5 rounded border border-slate-700 text-slate-400 hover:border-rose-600 hover:text-rose-300"
        onclick={resetConversation}
        {disabled}
        title="Discard all blocks and raw edits and start over with an empty user turn."
      >reset</button>
    </div>

    {#if subMode === 'blocks'}
      <ChatBlockList
        {doc}
        {profile}
        {backend}
        {model}
        {tokenizeSupported}
        {disabled}
        onChange={(d) => (doc = d)}
      />
      {#if parseWarnings.length}
        <div class="text-[11px] text-amber-400 space-y-0.5">
          {#each parseWarnings as w (w)}
            <div>⚠ {w}</div>
          {/each}
        </div>
      {/if}
    {:else}
      <ChatRawEditor
        value={subMode === 'raw' && rawEditable ? rawText : (rendered?.raw ?? '')}
        {backend}
        {model}
        {tokenizeSupported}
        {snippets}
        readonly={!rawEditable}
        {disabled}
        {parseError}
        onInput={(v) => {
          rawText = v;
          parseError = null;
        }}
        onDiscard={discardRawEdits}
      />
    {/if}

    {#if rendered?.error}
      <div class="rounded border border-rose-700/50 bg-rose-950/30 px-2.5 py-2 text-xs text-rose-300 leading-snug">
        <span class="font-semibold">the template itself rejected this conversation:</span>
        <span class="font-mono">{rendered.error}</span>
        <span class="block mt-1 text-rose-400/80">
          This is the Jinja program's own validation (e.g. Gemma has no
          system role; Mistral requires strict user/assistant alternation).
          Fix the blocks, or study why the family forbids it.
        </span>
      </div>
    {/if}
    {#if rendered && rendered.warnings.length && !info.is_base_model}
      <div class="text-[11px] text-amber-400 space-y-0.5">
        {#each rendered.warnings as w (w)}
          <div>⚠ {w}</div>
        {/each}
      </div>
    {/if}

    <ChatTemplatePanel {info} />
  {/if}
</div>
