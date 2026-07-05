<script lang="ts">
  /**
   * The editable token-highlighting primitive: a textarea with an inline
   * backdrop that shades token boundaries directly behind the text being
   * typed (alternating greys, specials in magenta), driven by a debounced
   * ``/api/v1/tokenize`` call.
   *
   * This is the single source of the "highlight in the input field"
   * behaviour: Text mode's prompt (via ``TokenComposer``, which adds the
   * special-token palette on top), chat Raw mode, and every chat block
   * card all render THIS component, so tokens look identical everywhere.
   *
   * The backdrop always renders text sliced from the live ``value`` (never
   * from the pieces directly), so the overlay stays pixel-aligned with the
   * textarea even when a tokenizer's per-piece decode doesn't perfectly
   * reconstruct the source string. With ``enabled=false`` (no local
   * tokenizer) it degrades to a plain textarea.
   */
  import { onDestroy, tick } from 'svelte';
  import {
    requestTokenize,
    segmentPieces,
    tokenizeErrorText,
    type TokenizePreview,
    type TokenizeSegment
  } from '$lib/tokenize';

  interface Props {
    /** Bound text being composed. */
    value: string;
    backend: string;
    model: string;
    /** False = no real local tokenizer: plain textarea, no meta row. */
    enabled: boolean;
    placeholder?: string;
    rows?: number;
    debounceMs?: number;
    /** Optional label on the left of the meta row (uppercase, muted). */
    label?: string;
    disabled?: boolean;
  }

  let {
    value = $bindable(),
    backend,
    model,
    enabled,
    placeholder = '',
    rows = 6,
    debounceMs = 200,
    label = '',
    disabled = false
  }: Props = $props();

  // ---- live tokenization (debounced + abortable) ---------------------- //
  let preview = $state<TokenizePreview | null>(null);
  let busy = $state(false);
  let tokError = $state('');
  let previewKey = $state('');
  let pending: ReturnType<typeof setTimeout> | null = null;
  let abortCtrl: AbortController | null = null;

  function inputKey(): string {
    return `${backend}|${model}|${value}`;
  }

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

  async function runTokenize(snapshot: string, b: string, m: string): Promise<void> {
    if (!enabled || !b) return;
    if (snapshot === '') {
      preview = { ids: [], pieces: [] };
      previewKey = `${b}|${m}|${snapshot}`;
      tokError = '';
      return;
    }
    clearPending();
    busy = true;
    tokError = '';
    const ctrl = new AbortController();
    abortCtrl = ctrl;
    try {
      const data = await requestTokenize(b, m, snapshot, ctrl.signal);
      if (snapshot !== value || b !== backend || m !== model) return;
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
    const snapshot = value;
    const b = backend;
    const m = model;
    const e = enabled;
    if (!e) {
      clearPending();
      preview = null;
      tokError = '';
      return;
    }
    if (pending !== null) clearTimeout(pending);
    pending = setTimeout(() => {
      pending = null;
      void runTokenize(snapshot, b, m);
    }, debounceMs);
  });

  onDestroy(() => clearPending());

  // ---- backdrop segmentation (always covers the exact value) ---------- //
  let isFresh = $derived(previewKey === inputKey());
  let tokenCount = $derived(preview ? preview.ids.length : 0);
  let segments = $derived.by<TokenizeSegment[]>(() =>
    preview && isFresh ? segmentPieces(value, preview.pieces) : []
  );

  // ---- caret-aware insertion ------------------------------------------ //
  let taEl = $state<HTMLTextAreaElement | null>(null);
  let backdropEl = $state<HTMLDivElement | null>(null);

  /** Splice ``s`` at the caret (palette chips, snippet buttons).
   * ``cursorBack`` parks the cursor that many chars earlier -- inside an
   * open/close marker pair. */
  export function insertText(s: string, cursorBack = 0): void {
    const el = taEl;
    const start = el ? el.selectionStart : value.length;
    const end = el ? el.selectionEnd : value.length;
    value = value.slice(0, start) + s + value.slice(end);
    void tick().then(() => {
      if (!el) return;
      const pos = start + s.length - cursorBack;
      el.focus();
      el.setSelectionRange(pos, pos);
      syncScroll();
    });
  }

  /** Focus + select a range (public: "jump to parse error"). */
  export function focusRange(start: number, end: number): void {
    const el = taEl;
    if (!el) return;
    el.focus();
    el.setSelectionRange(start, end);
  }

  function syncScroll(): void {
    if (taEl && backdropEl) {
      backdropEl.scrollTop = taEl.scrollTop;
      backdropEl.scrollLeft = taEl.scrollLeft;
    }
  }
</script>

<div class="tta">
  {#if enabled && (label || preview || busy || tokError)}
    <div class="tta-head">
      <span class="tta-label">{label}</span>
      <span class="meta">
        {#if busy}<span class="dot">…</span>{/if}
        <span class="count">{tokenCount} {tokenCount === 1 ? 'token' : 'tokens'}</span>
        {#if !isFresh && preview && !busy}
          <span class="stale" title="recomputing tokens for the latest text">stale</span>
        {/if}
        {#if tokError}<span class="err" title={tokError}>{tokError}</span>{/if}
      </span>
    </div>
  {/if}

  {#if enabled}
    <div class="editor">
      <div class="backdrop" bind:this={backdropEl} aria-hidden="true">{#if segments.length}{#each segments as seg (seg.idx)}<span
              class="tok {seg.special ? 'tok-special' : seg.idx % 2 === 0 ? 'tok-a' : 'tok-b'}"
              >{seg.text}</span
            >{/each}{:else}<span class="tok-plain">{value}</span>{/if}{#if value.endsWith('\n')}<span>&nbsp;</span>{/if}</div>
      <textarea
        bind:this={taEl}
        bind:value
        {rows}
        {placeholder}
        {disabled}
        spellcheck="false"
        class="ta"
        onscroll={syncScroll}
      ></textarea>
    </div>
  {:else}
    <textarea bind:value {rows} {disabled} {placeholder} spellcheck="false" class="ta ta-plain"></textarea>
  {/if}
</div>

<style>
  .tta {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
  }
  .tta-head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-size: 0.72rem;
    min-height: 1em;
  }
  .tta-label {
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8;
  }
  .meta {
    display: inline-flex;
    gap: 0.45rem;
    align-items: baseline;
    color: #94a3b8;
  }
  .count {
    font-variant-numeric: tabular-nums;
    color: #cbd5e1;
  }
  .stale {
    color: #b58000;
    font-style: italic;
  }
  .dot {
    color: #64748b;
  }
  .err {
    color: #f87171;
  }

  /* The overlay: backdrop + textarea share identical box metrics so the
     coloured token spans sit exactly behind the glyphs. */
  .editor {
    position: relative;
  }
  .backdrop,
  .ta {
    margin: 0;
    border: 1px solid rgb(51 65 85);
    border-radius: 0.375rem;
    padding: 0.5rem 0.625rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.85rem;
    line-height: 1.5;
    white-space: pre-wrap;
    overflow-wrap: break-word;
    word-break: break-word;
    box-sizing: border-box;
    width: 100%;
  }
  .backdrop {
    position: absolute;
    inset: 0;
    overflow: auto;
    pointer-events: none;
    color: transparent;
    background: rgb(15 23 42);
    z-index: 0;
  }
  .ta {
    position: relative;
    z-index: 1;
    background: transparent;
    color: rgb(226 232 240);
    caret-color: rgb(56 189 248);
    resize: vertical;
    display: block;
  }
  .ta::placeholder {
    color: #64748b;
  }
  .ta-plain {
    background: rgb(15 23 42);
  }
  .ta:focus,
  .ta-plain:focus {
    outline: none;
    border-color: rgb(56 189 248);
  }
  /* Token backgrounds: pure background only (no padding/margin) so the
     overlay never shifts a single glyph. Alternating shades make boundaries
     legible; specials pop in magenta. */
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
    color: transparent;
    box-shadow: 0 0 0 1px rgba(217, 70, 239, 0.55) inset;
  }
  .tok-plain {
    color: transparent;
  }
</style>
