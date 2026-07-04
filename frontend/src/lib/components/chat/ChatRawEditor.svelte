<script lang="ts">
  /**
   * Raw sub-mode: direct textarea editing of the rendered template text,
   * with model-correct snippet buttons (generated from the derived
   * TemplateProfile -- see lib/chat/snippets.ts) and the positioned
   * parse-error display for the Raw -> Blocks switch.
   *
   * In simulation mode (chat-only providers) the same component renders
   * read-only: the provider applies the real template server-side, so the
   * text shown is our best local reconstruction, not what we send.
   */
  import type { ParseError, Snippet } from '$lib/chat/types';

  interface Props {
    value: string;
    snippets: Snippet[];
    readonly?: boolean;
    disabled?: boolean;
    parseError?: ParseError | null;
    onInput: (value: string) => void;
  }
  let {
    value,
    snippets,
    readonly = false,
    disabled = false,
    parseError = null,
    onInput
  }: Props = $props();

  let textarea = $state<HTMLTextAreaElement | null>(null);

  /** Insert a snippet at the caret, then park the cursor ``cursorBack``
   * chars earlier (inside an open/close marker pair). */
  function insertSnippet(s: Snippet): void {
    const el = textarea;
    if (!el) return;
    const start = el.selectionStart ?? value.length;
    const end = el.selectionEnd ?? start;
    const next = value.slice(0, start) + s.text + value.slice(end);
    onInput(next);
    requestAnimationFrame(() => {
      el.focus();
      const caret = start + s.text.length - s.cursorBack;
      el.setSelectionRange(caret, caret);
    });
  }

  /** Line/column of the parse-error position, for the human-readable
   * message (character offsets alone are hard to act on in a textarea). */
  let errorLineCol = $derived.by<{ line: number; col: number } | null>(() => {
    if (!parseError) return null;
    const upto = value.slice(0, parseError.position);
    const lines = upto.split('\n');
    return { line: lines.length, col: lines[lines.length - 1].length + 1 };
  });

  function focusError(): void {
    const el = textarea;
    if (!el || !parseError) return;
    el.focus();
    el.setSelectionRange(parseError.position, Math.min(parseError.position + 8, value.length));
  }
</script>

<div class="space-y-1.5">
  {#if snippets.length && !readonly}
    <div class="flex flex-wrap gap-1">
      {#each snippets as s (s.id)}
        <button
          type="button"
          class="text-[11px] px-1.5 py-0.5 rounded border border-slate-700 font-mono hover:border-sky-600 hover:text-sky-300 disabled:opacity-40"
          title={s.title}
          {disabled}
          onclick={() => insertSnippet(s)}
        >{s.label}</button>
      {/each}
    </div>
  {/if}

  <textarea
    bind:this={textarea}
    class="input w-full font-mono text-xs leading-relaxed {readonly ? 'opacity-70' : ''}"
    rows="10"
    spellcheck="false"
    {readonly}
    {disabled}
    {value}
    oninput={(e) => onInput(e.currentTarget.value)}
  ></textarea>

  {#if parseError}
    <div class="rounded border border-rose-700/50 bg-rose-950/30 px-2.5 py-2 text-xs text-rose-300 leading-snug space-y-1">
      <div>
        <span class="font-semibold">parse error</span>
        {#if errorLineCol}
          at line {errorLineCol.line}, column {errorLineCol.col}
          <button
            type="button"
            class="ml-1 underline decoration-dotted hover:text-rose-100"
            onclick={focusError}
          >jump to it</button>
        {/if}
      </div>
      <div>expected {parseError.expected}</div>
      <div>found <span class="font-mono">{parseError.found}</span></div>
      <div class="text-rose-400/90">{parseError.hint}</div>
    </div>
  {/if}
</div>
