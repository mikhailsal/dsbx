<script lang="ts">
  /**
   * Raw sub-mode: direct editing of the rendered template text through the
   * SAME token-aware composer as Text mode -- inline token-boundary
   * highlighting + the model's special-token palette -- plus the
   * model-correct snippet buttons (generated from the derived
   * TemplateProfile, see lib/chat/snippets.ts) and the positioned
   * parse-error display for the Raw -> Blocks switch.
   *
   * In simulation mode (chat-only providers) and for templates the parser
   * cannot round-trip, the same component renders a READ-ONLY tokenized
   * view instead: the provider applies the real template server-side, so
   * the text shown is our best local reconstruction, not what we send.
   */
  import TokenComposer from '$lib/components/TokenComposer.svelte';
  import TokenizedText from '$lib/components/TokenizedText.svelte';
  import type { ParseError, Snippet } from '$lib/chat/types';

  interface Props {
    value: string;
    backend: string;
    model: string;
    /** Backend has a real local tokenizer (highlighting possible). */
    tokenizeSupported: boolean;
    snippets: Snippet[];
    readonly?: boolean;
    disabled?: boolean;
    parseError?: ParseError | null;
    onInput: (value: string) => void;
    /** Escape hatch shown with the parse error: discard the raw edits
     * and return to the (still intact) blocks. */
    onDiscard?: (() => void) | null;
  }
  let {
    value,
    backend,
    model,
    tokenizeSupported,
    snippets,
    readonly = false,
    disabled = false,
    parseError = null,
    onInput,
    onDiscard = null
  }: Props = $props();

  let composer = $state<TokenComposer | null>(null);

  /** Line/column of the parse-error position, for the human-readable
   * message (character offsets alone are hard to act on in a textarea). */
  let errorLineCol = $derived.by<{ line: number; col: number } | null>(() => {
    if (!parseError) return null;
    const upto = value.slice(0, parseError.position);
    const lines = upto.split('\n');
    return { line: lines.length, col: lines[lines.length - 1].length + 1 };
  });
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
          onclick={() => composer?.insertText(s.text, s.cursorBack)}
        >{s.label}</button>
      {/each}
    </div>
  {/if}

  {#if readonly}
    <TokenizedText text={value} {backend} {model} enabled={tokenizeSupported} />
  {:else}
    <TokenComposer
      bind:this={composer}
      bind:value={() => value, (v) => onInput(v)}
      {backend}
      {model}
      enabled={tokenizeSupported}
      {disabled}
      rows={10}
      label="raw prompt"
      placeholder="The rendered template text — edit freely; special tokens round-trip to single ids."
    />
  {/if}

  {#if parseError}
    <div class="rounded border border-rose-700/50 bg-rose-950/30 px-2.5 py-2 text-xs text-rose-300 leading-snug space-y-1">
      <div>
        <span class="font-semibold">parse error</span>
        {#if errorLineCol}
          at line {errorLineCol.line}, column {errorLineCol.col}
          <button
            type="button"
            class="ml-1 underline decoration-dotted hover:text-rose-100"
            onclick={() =>
              parseError &&
              composer?.focusRange(
                parseError.position,
                Math.min(parseError.position + 8, value.length)
              )}
          >jump to it</button>
        {/if}
      </div>
      <div>expected {parseError.expected}</div>
      <div>found <span class="font-mono">{parseError.found}</span></div>
      <div class="text-rose-400/90">{parseError.hint}</div>
      {#if onDiscard}
        <div>
          You can keep editing here (Raw stays fully usable), or
          <button
            type="button"
            class="underline decoration-dotted hover:text-rose-100"
            onclick={onDiscard}
          >discard the raw edits and go back to Blocks</button>.
        </div>
      {/if}
    </div>
  {/if}
</div>
