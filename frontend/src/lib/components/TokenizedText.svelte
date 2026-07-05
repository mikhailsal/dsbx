<script lang="ts">
  /**
   * Read-only token-boundary highlighting for an arbitrary text: the
   * chat-mode sibling of ``TokenComposer``'s backdrop. Used for the
   * rendered-prompt preview in Blocks mode and the simulation-mode raw
   * preview, so "what the model actually sees" is shown with the same
   * alternating token shading (specials in magenta) as the editable
   * prompt in Text mode.
   *
   * Degrades to plain text when the backend has no local tokenizer
   * (``enabled=false``) or while the first tokenize is in flight -- it
   * never blocks showing the text itself.
   */
  import { onDestroy } from 'svelte';
  import {
    requestTokenize,
    segmentPieces,
    tokenizeErrorText,
    type TokenizePreview,
    type TokenizeSegment
  } from '$lib/tokenize';

  interface Props {
    text: string;
    backend: string;
    model: string;
    enabled: boolean;
    debounceMs?: number;
  }
  let { text, backend, model, enabled, debounceMs = 250 }: Props = $props();

  let preview = $state<TokenizePreview | null>(null);
  let previewKey = $state('');
  let busy = $state(false);
  let tokError = $state('');
  let pending: ReturnType<typeof setTimeout> | null = null;
  let abortCtrl: AbortController | null = null;

  function clearPending(): void {
    if (pending !== null) {
      clearTimeout(pending);
      pending = null;
    }
    if (abortCtrl) {
      abortCtrl.abort();
      abortCtrl = null;
    }
  }

  async function run(snapshot: string, b: string, m: string): Promise<void> {
    if (snapshot === '') {
      preview = { ids: [], pieces: [] };
      previewKey = `${b}|${m}|`;
      tokError = '';
      return;
    }
    busy = true;
    tokError = '';
    const ctrl = new AbortController();
    abortCtrl = ctrl;
    try {
      const data = await requestTokenize(b, m, snapshot, ctrl.signal);
      if (snapshot !== text || b !== backend || m !== model) return;
      preview = data;
      previewKey = `${b}|${m}|${snapshot}`;
    } catch (err) {
      if ((err as Error).name === 'AbortError') return;
      tokError = tokenizeErrorText(err);
    } finally {
      if (abortCtrl === ctrl) abortCtrl = null;
      busy = false;
    }
  }

  $effect(() => {
    const snapshot = text;
    const b = backend;
    const m = model;
    if (!enabled || !b) {
      clearPending();
      preview = null;
      tokError = '';
      return;
    }
    if (pending !== null) clearTimeout(pending);
    pending = setTimeout(() => {
      pending = null;
      void run(snapshot, b, m);
    }, debounceMs);
  });

  onDestroy(() => clearPending());

  let isFresh = $derived(previewKey === `${backend}|${model}|${text}`);
  let tokenCount = $derived(preview ? preview.ids.length : 0);
  let segments = $derived.by<TokenizeSegment[]>(() =>
    preview && isFresh ? segmentPieces(text, preview.pieces) : []
  );
</script>

<div class="tt">
  {#if enabled}
    <div class="tt-head">
      {#if busy}<span class="dot">…</span>{/if}
      {#if isFresh && preview}<span class="count"
          >{tokenCount} {tokenCount === 1 ? 'token' : 'tokens'}</span
        >{/if}
      {#if tokError}<span class="err" title={tokError}>{tokError}</span>{/if}
    </div>
  {/if}
  <div class="tt-body">{#if segments.length}{#each segments as seg (seg.idx)}<span
          class="tok {seg.special ? 'tok-special' : seg.idx % 2 === 0 ? 'tok-a' : 'tok-b'}"
          >{seg.text}</span
        >{/each}{:else}{text}{/if}</div>
</div>

<style>
  .tt-head {
    display: flex;
    justify-content: flex-end;
    gap: 0.45rem;
    font-size: 0.68rem;
    color: #94a3b8;
    min-height: 1em;
  }
  .count {
    font-variant-numeric: tabular-nums;
    color: #cbd5e1;
  }
  .dot {
    color: #64748b;
  }
  .err {
    color: #f87171;
  }
  .tt-body {
    border: 1px solid rgb(51 65 85);
    border-radius: 0.375rem;
    padding: 0.5rem 0.625rem;
    background: rgb(15 23 42);
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.78rem;
    line-height: 1.6;
    white-space: pre-wrap;
    overflow-wrap: break-word;
    word-break: break-word;
    color: rgb(226 232 240);
    max-height: 20rem;
    overflow-y: auto;
  }
  .tok {
    border-radius: 2px;
  }
  .tok-a {
    background: rgba(148, 163, 184, 0.1);
  }
  .tok-b {
    background: rgba(148, 163, 184, 0.22);
  }
  .tok-special {
    background: rgba(217, 70, 239, 0.35);
    color: #f5d0fe;
    box-shadow: 0 0 0 1px rgba(217, 70, 239, 0.55) inset;
  }
</style>
