/**
 * Watch-column reconstruction, extracted from
 * ``routes/generate/+page.svelte``. The frontend rebuilds human-readable
 * watch column headers from what IT sent (no server-side ResolvedWatch
 * round-trip). De-duplication semantics mirror the server's
 * ``_resolve_watches`` so the columns line up with the per-step
 * ``watched`` arrays.
 */

import type { StepResult, TokenCandidate } from '$lib/types';

export interface WatchColumn {
  label: string;
  tokenId: number;
  source: 'text' | 'id' | 'eos';
}

/**
 * Build the column headers for the current watch configuration. Text
 * watches can't be pre-resolved to ids in-browser (no tokenizer), so
 * they render as placeholder columns matched positionally -- text
 * watches always lead the server's resolved order.
 */
export function buildWatchColumns(opts: {
  watchTexts: string[];
  watchIds: string[];
  watchEos: boolean;
  eosTokenIds: number[];
  tokenCache: Record<number, string>;
}): WatchColumn[] {
  const out: WatchColumn[] = [];
  const seen = new Set<number>();
  for (const t of opts.watchTexts) {
    out.push({ label: `text:${JSON.stringify(t)}`, tokenId: -1, source: 'text' });
  }
  for (const raw of opts.watchIds) {
    const tid = Number.parseInt(raw, 10);
    if (!Number.isFinite(tid) || seen.has(tid)) continue;
    seen.add(tid);
    const piece = opts.tokenCache[tid];
    const suffix = piece ? ` ${JSON.stringify(piece)}` : '';
    out.push({ label: `id=${tid}${suffix}`, tokenId: tid, source: 'id' });
  }
  if (opts.watchEos) {
    for (const tid of opts.eosTokenIds) {
      if (seen.has(tid)) continue;
      seen.add(tid);
      out.push({ label: `EOS:${tid}`, tokenId: tid, source: 'eos' });
    }
  }
  return out;
}

/**
 * Resolve one watch column at a given position by looking it up in the
 * row's ``watched`` array. ``text`` columns use the positional index
 * trick (text watches always come first in the server's resolved
 * order); ``id`` / ``eos`` columns look their id up directly.
 */
export function watchedAt(
  step: StepResult,
  col: WatchColumn,
  textIdxIfApplicable: number
): TokenCandidate | null {
  if (col.source === 'text') {
    const w = step.watched[textIdxIfApplicable];
    return w ? w.candidate : null;
  }
  const w = step.watched.find((x) => x.token_id === col.tokenId);
  return w ? w.candidate : null;
}
