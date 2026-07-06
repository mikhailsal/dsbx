/**
 * Pure request-building logic for the Decode workbench, extracted from
 * ``routes/generate/+page.svelte`` so it can be unit-tested and shared
 * (the chat composer and any future page reuse the exact same wire
 * assembly). No Svelte state in here -- everything comes in through the
 * config object and the result is a plain JSON-ready body.
 */

export type SamplerName =
  | 'greedy'
  | 'temperature'
  | 'top_k'
  | 'top_p'
  | 'min_p'
  | 'typical'
  | 'mirostat';

/** Every sampler knob the UI exposes; ``buildSamplerParams`` picks the
 * subset the selected sampler actually uses. */
export interface SamplerKnobs {
  temperature: number;
  topK: number;
  topP: number;
  minP: number;
  typicalP: number;
  mirostatTarget: number;
  mirostatLr: number;
  repetitionPenalty: number;
  frequencyPenalty: number;
  presencePenalty: number;
}

/**
 * Wire params for the selected sampler. Penalties ride along on every
 * sampler but only when they differ from their no-op defaults, so the
 * request body stays minimal (and the server's "unsupported field"
 * advisories only fire when the user actually opted in).
 */
export function buildSamplerParams(
  name: SamplerName,
  knobs: SamplerKnobs
): Record<string, number | null> {
  const penalties: Record<string, number> = {};
  if (knobs.repetitionPenalty !== 1.0) penalties.repetition_penalty = knobs.repetitionPenalty;
  if (knobs.frequencyPenalty !== 0.0) penalties.frequency_penalty = knobs.frequencyPenalty;
  if (knobs.presencePenalty !== 0.0) penalties.presence_penalty = knobs.presencePenalty;
  switch (name) {
    case 'greedy':
      return { ...penalties };
    case 'temperature':
      return { temperature: knobs.temperature, ...penalties };
    case 'top_k':
      return { temperature: knobs.temperature, top_k: knobs.topK, ...penalties };
    case 'top_p':
      return { temperature: knobs.temperature, top_p: knobs.topP, ...penalties };
    case 'min_p':
      return { temperature: knobs.temperature, min_p: knobs.minP, ...penalties };
    case 'typical':
      return { temperature: knobs.temperature, typical_p: knobs.typicalP, ...penalties };
    case 'mirostat':
      return {
        temperature: knobs.temperature,
        mirostat_target: knobs.mirostatTarget,
        mirostat_lr: knobs.mirostatLr,
        ...penalties
      };
  }
}

export interface LogitBiasRow {
  id: string;
  tokenId: string;
  bias: string;
}

/**
 * Convert the editor rows to the wire shape ``{<token_id>: <bias>}``
 * with string keys (JSON requirement). Silently drop rows with
 * unparseable ids or out-of-range biases; the backend would reject
 * those anyway and we'd rather the user keep their partial input
 * visible than have the page nuke half their edits at submit time.
 * Returns ``undefined`` -- not an empty object -- when there are no
 * valid rows, so the request omits the field entirely.
 */
export function collectLogitBias(
  rows: LogitBiasRow[],
  supported: boolean
): Record<string, number> | undefined {
  if (!supported) return undefined;
  const out: Record<string, number> = {};
  for (const row of rows) {
    const tid = Number.parseInt(row.tokenId, 10);
    const bias = Number.parseFloat(row.bias);
    if (!Number.isFinite(tid) || !Number.isFinite(bias)) continue;
    if (bias < -100 || bias > 100) continue;
    out[String(tid)] = bias;
  }
  return Object.keys(out).length ? out : undefined;
}

/** Parse a chip-input's string values into finite integer token ids
 * (shared by stop ids, watch ids, and prepend ids). */
export function parseIdChips(values: string[]): number[] {
  return values.map((s) => Number.parseInt(s, 10)).filter((n) => Number.isFinite(n));
}

/** Everything ``buildGenerateRequest`` needs, gathered from page state.
 * Prompt-source fields (``prompt`` / ``messages`` / ``tools``) are
 * already mode-resolved by the caller -- this module doesn't know about
 * the Text|Chat toggle, only about the wire shape. */
export interface GenerateRequestConfig {
  backend: string;
  model: string;
  prompt: string;
  /** Structured conversation for the chat-simulation path; null for
   * the raw-prompt path (the field is then omitted from the wire). */
  messages: Record<string, unknown>[] | null;
  tools: Record<string, unknown>[] | null;
  sampler: SamplerName;
  samplerKnobs: SamplerKnobs;
  maxTokens: number;
  alternatives: number;
  stopTexts: string[];
  stopIds: string[];
  seed: number;
  respectEos: boolean;
  includePrompt: boolean;
  serviceTier: 'default' | 'priority';
  serviceTierSupported: boolean;
  logitBiasRows: LogitBiasRow[];
  logitBiasSupported: boolean;
  echoLast: number;
  combinedEchoStreamSupported: boolean;
  watchTexts: string[];
  watchIds: string[];
  watchEos: boolean;
  prependTokenIds: string[];
  prependSupported: boolean;
}

/** Per-call overrides layered on top of the shared config so the three
 * action buttons (inspect / generate / manual) stay one-liners. */
export interface GenerateRequestOverrides {
  maxTokensOverride?: number;
  includePromptOverride?: boolean;
  /** Manual picker's accumulated picks; ``[]`` for inspect/generate. */
  prefix?: number[];
  forManual?: boolean;
  manualSessionId?: string;
  manualCacheKey?: string;
}

export interface BuiltGenerateRequest {
  body: Record<string, unknown>;
  /** How many prepend ids were actually sent -- pinned at request time
   * so the prompt-logits table can highlight the BOS-conditioned row
   * even after the user edits the chip-input between runs. */
  prependCount: number;
}

/**
 * Build the request body for a generate-stream call. Pure: same
 * config + overrides in, same body out. The caller stores
 * ``prependCount`` for the table-highlight bookkeeping.
 */
export function buildGenerateRequest(
  cfg: GenerateRequestConfig,
  opts: GenerateRequestOverrides
): BuiltGenerateRequest {
  const includeP = opts.includePromptOverride ?? cfg.includePrompt;
  const prependIds = cfg.prependSupported ? parseIdChips(cfg.prependTokenIds) : [];
  const body: Record<string, unknown> = {
    backend: cfg.backend,
    model: cfg.model || undefined,
    prompt: cfg.prompt,
    messages: cfg.messages ?? undefined,
    tools: cfg.tools?.length ? cfg.tools : undefined,
    sampler: { name: cfg.sampler, params: buildSamplerParams(cfg.sampler, cfg.samplerKnobs) },
    max_tokens: opts.maxTokensOverride ?? cfg.maxTokens,
    top_k: cfg.alternatives,
    stop_texts: cfg.stopTexts,
    stop_ids: parseIdChips(cfg.stopIds),
    seed: cfg.seed,
    respect_eos: cfg.respectEos,
    include_prompt: includeP,
    service_tier: cfg.serviceTierSupported ? cfg.serviceTier : undefined,
    logit_bias: collectLogitBias(cfg.logitBiasRows, cfg.logitBiasSupported),
    echo_last:
      includeP && cfg.combinedEchoStreamSupported && cfg.echoLast > 0 ? cfg.echoLast : undefined,
    watch_texts: cfg.watchTexts,
    watch_ids: parseIdChips(cfg.watchIds),
    watch_eos: cfg.watchEos,
    prefix_token_ids: opts.prefix ?? [],
    prepend_token_ids: prependIds,
    // Manual mode pins both UUIDs so Fireworks can reuse the KV cache
    // and MoE expert routing across picks; for the other modes we
    // leave them ``undefined`` so each click is a fresh request.
    session_id: opts.forManual ? opts.manualSessionId : undefined,
    prompt_cache_key: opts.forManual ? opts.manualCacheKey : undefined
  };
  return { body, prependCount: prependIds.length };
}
