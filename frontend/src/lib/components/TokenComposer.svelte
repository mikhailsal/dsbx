<script lang="ts">
  /**
   * Token-aware prompt composer for the Decode workbench: the shared
   * ``TokenTextarea`` (inline token-boundary highlighting directly in the
   * editable field) plus the model-specific affordances that only the
   * full prompt editor needs:
   *
   *   1. A palette of EVERY special token the tokenizer knows (BOS / EOS /
   *      chat / tool markers), fetched from ``/api/v1/special_tokens``.
   *      Clicking one splices its exact string at the caret; because the
   *      backend tokenizes with special-token matching on, that string
   *      round-trips to the single control-token id it names. This is how
   *      "condition on BOS" works: insert the BOS token at the start of
   *      the prompt -- no separate prepend field needed.
   *   2. Insert an arbitrary token BY ID: we resolve the id to its surface
   *      text via ``/api/v1/piece`` and splice that. (Plain non-special
   *      pieces are re-tokenized on the server, so this is "usually exact"
   *      rather than guaranteed -- special tokens, which DO round-trip,
   *      are the precise path.)
   */
  import { apiFetch, ApiError } from '$lib/api';
  import TokenTextarea from '$lib/components/TokenTextarea.svelte';

  interface Props {
    /** Bound prompt text (the model input being composed). */
    value: string;
    backend: string;
    model: string;
    /**
     * When false the backend has no real local tokenizer: we fall back to
     * a plain textarea (no highlight, no palette) so the field still works
     * but never lies with a one-chip-for-everything stub.
     */
    enabled: boolean;
    placeholder?: string;
    rows?: number;
    debounceMs?: number;
    /** Header label ("prompt" on the Decode page, "raw prompt" in chat). */
    label?: string;
    disabled?: boolean;
  }

  let {
    value = $bindable(),
    backend,
    model,
    enabled,
    placeholder = 'Type your prompt. Insert special tokens from the palette below.',
    rows = 6,
    debounceMs = 200,
    label = 'prompt',
    disabled = false
  }: Props = $props();

  let editor = $state<TokenTextarea | null>(null);

  /** Splice ``s`` at the caret (public: chat raw mode's snippet buttons). */
  export function insertText(s: string, cursorBack = 0): void {
    editor?.insertText(s, cursorBack);
  }

  /** Focus + select a range (public: "jump to parse error"). */
  export function focusRange(start: number, end: number): void {
    editor?.focusRange(start, end);
  }

  // ---- special-token palette ----------------------------------------- //
  interface SpecialTok {
    id: number;
    text: string;
  }
  let specials = $state<SpecialTok[]>([]);
  let specialsErr = $state('');
  let specialsKey = $state('');
  let paletteOpen = $state(true);
  let search = $state('');
  let showNoise = $state(false);
  const MAX_PALETTE = 80;

  // Some tokenizers pad their special vocab with hundreds/thousands of
  // inert markers -- DeepSeek ships 800 ``<｜place▁holder▁no▁N｜>`` + 415
  // ``<|place_holder_mm_span_N|>`` (1215 of 1230!), gpt-oss has
  // ``<|reserved_NNN|>``. These bury the ~15 genuinely useful tokens past
  // the visible cap. We bucket them as "noise" and hide them by default
  // (a toggle reveals them; search always covers the FULL set so nothing
  // is permanently unreachable).
  function isNoiseSpecial(text: string): boolean {
    const norm = text.replace(/[<>|\uFF5C\u2581_ .]/g, '').toLowerCase();
    return /(placeholder|reserved|unused)/.test(norm);
  }

  $effect(() => {
    const b = backend;
    const m = model;
    if (!enabled || !b) {
      specials = [];
      return;
    }
    const key = `${b}|${m}`;
    if (key === specialsKey) return;
    specialsKey = key;
    specialsErr = '';
    void apiFetch<{ tokens: SpecialTok[] }>('/api/v1/special_tokens', {
      method: 'POST',
      body: JSON.stringify({ backend: b, model: m })
    })
      .then((r) => {
        if (`${backend}|${model}` !== key) return;
        specials = r.tokens ?? [];
      })
      .catch((err) => {
        if (`${backend}|${model}` !== key) return;
        specials = [];
        specialsErr =
          err instanceof ApiError ? `HTTP ${err.status}` : (err as Error).message;
      });
  });

  let usefulSpecials = $derived<SpecialTok[]>(
    specials.filter((s) => !isNoiseSpecial(s.text))
  );
  let noiseCount = $derived(specials.length - usefulSpecials.length);

  let filteredSpecials = $derived.by<SpecialTok[]>(() => {
    const q = search.trim().toLowerCase();
    // While searching, search the FULL set (incl. noise) so a known
    // reserved/placeholder id is still reachable; otherwise show useful
    // only unless the user opted into the noise via the toggle.
    const base = q ? specials : showNoise ? specials : usefulSpecials;
    if (!q) return base;
    return base.filter(
      (s) => s.text.toLowerCase().includes(q) || String(s.id).includes(q)
    );
  });
  let shownSpecials = $derived(filteredSpecials.slice(0, MAX_PALETTE));

  // ---- insert by id --------------------------------------------------- //
  let idInput = $state('');
  let idErr = $state('');

  async function insertById(): Promise<void> {
    idErr = '';
    const id = Number.parseInt(idInput, 10);
    if (!Number.isFinite(id)) {
      idErr = 'enter a numeric id';
      return;
    }
    try {
      const r = await apiFetch<{ text: string }>('/api/v1/piece', {
        method: 'POST',
        body: JSON.stringify({ backend, model, id })
      });
      if (!r.text) {
        idErr = `id ${id} has no surface text`;
        return;
      }
      editor?.insertText(r.text);
      idInput = '';
    } catch (err) {
      idErr = err instanceof ApiError ? `HTTP ${err.status}` : (err as Error).message;
    }
  }
</script>

<div class="composer">
  <TokenTextarea
    bind:this={editor}
    bind:value
    {backend}
    {model}
    {enabled}
    {rows}
    {debounceMs}
    {label}
    {disabled}
    placeholder={enabled ? placeholder : 'prompt (no local tokenizer on this backend)'}
  />

  {#if enabled}
    <div class="palette">
      <button
        type="button"
        class="palette-toggle"
        onclick={() => (paletteOpen = !paletteOpen)}
        aria-expanded={paletteOpen}
      >
        {paletteOpen ? '▾' : '▸'} special tokens
        <span class="palette-count">{usefulSpecials.length}</span>
        {#if specialsErr}<span class="err">({specialsErr})</span>{/if}
      </button>
      {#if paletteOpen && noiseCount > 0}
        <button
          type="button"
          class="noise-toggle"
          onclick={() => (showNoise = !showNoise)}
          title="Reserved / placeholder / unused markers the tokenizer pads its vocab with. They're hidden by default because they bury the useful tokens; search always finds them regardless."
        >
          {showNoise ? 'hide' : 'show'} {noiseCount} reserved/placeholder
        </button>
      {/if}

      {#if paletteOpen}
        {#if specials.length === 0}
          <p class="palette-empty">
            {specialsErr
              ? 'could not load special tokens for this model'
              : 'this model exposes no special tokens'}
          </p>
        {:else}
          <div class="palette-controls">
            <input
              type="text"
              class="palette-search"
              placeholder="filter by name or id…"
              bind:value={search}
            />
            <div class="by-id">
              <input
                type="text"
                inputmode="numeric"
                class="id-input"
                placeholder="id"
                bind:value={idInput}
                onkeydown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    void insertById();
                  }
                }}
              />
              <button type="button" class="id-btn" onclick={() => void insertById()}>
                insert by id
              </button>
            </div>
          </div>
          {#if idErr}<p class="err id-err">{idErr}</p>{/if}
          <div class="chips">
            {#each shownSpecials as s (s.id)}
              <button
                type="button"
                class="chip"
                title={`insert ${s.text} (id ${s.id})`}
                onclick={() => editor?.insertText(s.text)}
              >
                <span class="chip-text">{s.text}</span>
                <span class="chip-id">{s.id}</span>
              </button>
            {/each}
            {#if filteredSpecials.length > shownSpecials.length}
              <span class="more">
                +{filteredSpecials.length - shownSpecials.length} more — refine the filter
              </span>
            {/if}
          </div>
        {/if}
        <p class="hint">
          Inserting a special token splices its exact string at the cursor; it
          tokenizes back to that single id. Put the model's BOS at the very start
          to condition generation on it (replaces the old “prepend BOS”).
        </p>
      {/if}
    </div>
  {/if}
</div>

<style>
  .composer {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }
  .err {
    color: #f87171;
  }

  /* palette */
  .palette {
    font-size: 0.75rem;
  }
  .palette-toggle {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    color: #cbd5e1;
    background: none;
    border: none;
    cursor: pointer;
    padding: 0.1rem 0;
  }
  .palette-count {
    font-variant-numeric: tabular-nums;
    color: #64748b;
    background: rgba(148, 163, 184, 0.15);
    border-radius: 999px;
    padding: 0 0.4rem;
  }
  .noise-toggle {
    margin-left: 0.5rem;
    background: none;
    border: none;
    color: #64748b;
    cursor: pointer;
    text-decoration: underline dotted;
    font-size: 0.72rem;
    padding: 0;
  }
  .noise-toggle:hover {
    color: #94a3b8;
  }
  .palette-empty {
    color: #64748b;
    font-style: italic;
    margin: 0.25rem 0;
  }
  .palette-controls {
    display: flex;
    gap: 0.4rem;
    margin: 0.35rem 0;
    flex-wrap: wrap;
  }
  .palette-search,
  .id-input {
    background: rgb(15 23 42);
    border: 1px solid rgb(51 65 85);
    border-radius: 0.375rem;
    padding: 0.25rem 0.45rem;
    color: rgb(226 232 240);
    font-size: 0.75rem;
  }
  .palette-search {
    flex: 1 1 12rem;
    min-width: 8rem;
  }
  .id-input {
    width: 5rem;
    font-family: ui-monospace, monospace;
  }
  .by-id {
    display: inline-flex;
    gap: 0.3rem;
  }
  .id-btn,
  .chip {
    background: rgba(148, 163, 184, 0.12);
    border: 1px solid rgb(51 65 85);
    border-radius: 0.375rem;
    color: #cbd5e1;
    cursor: pointer;
  }
  .id-btn {
    padding: 0.25rem 0.55rem;
    font-size: 0.72rem;
    white-space: nowrap;
  }
  .id-btn:hover {
    border-color: rgb(100 116 139);
  }
  .id-err {
    margin: 0.15rem 0;
  }
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
    max-height: 11rem;
    overflow-y: auto;
    padding: 0.15rem 0.1rem;
  }
  .chip {
    display: inline-flex;
    align-items: baseline;
    gap: 0.3rem;
    padding: 0.1rem 0.4rem;
    font-family: ui-monospace, monospace;
    font-size: 0.72rem;
  }
  .chip:hover {
    border-color: rgba(217, 70, 239, 0.6);
    background: rgba(217, 70, 239, 0.15);
  }
  .chip-text {
    color: #e9d5ff;
  }
  .chip-id {
    color: #64748b;
    font-size: 0.62rem;
    font-variant-numeric: tabular-nums;
  }
  .more {
    color: #64748b;
    font-style: italic;
    align-self: center;
  }
  .hint {
    color: #64748b;
    line-height: 1.4;
    margin: 0.35rem 0 0;
  }
</style>
