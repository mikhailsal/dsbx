/**
 * Shared live-tokenization plumbing for token-boundary highlighting.
 *
 * Used by ``TokenComposer`` (editable prompt with a highlight backdrop)
 * and ``TokenizedText`` (read-only highlighted view in chat mode) so both
 * render token boundaries from exactly the same segmentation rules.
 */

import { apiFetch, ApiError } from '$lib/api';
import { isSpecialText } from '$lib/render';

export interface TokenizePreview {
  ids: number[];
  pieces: string[];
}

export interface TokenizeSegment {
  text: string;
  special: boolean;
  idx: number;
}

/**
 * Attribute each piece's length to a slice OF THE SOURCE string, so the
 * concatenated segments always equal ``text`` -- this is what keeps a
 * highlight overlay pixel-aligned with a textarea even when a tokenizer's
 * per-piece decode doesn't perfectly reconstruct the source. When pieces
 * run short (or decode diverges) the remainder becomes one plain segment
 * rather than desyncing the whole field.
 */
export function segmentPieces(text: string, pieces: string[]): TokenizeSegment[] {
  if (pieces.length === 0) return [];
  const segs: TokenizeSegment[] = [];
  let pos = 0;
  for (let k = 0; k < pieces.length && pos <= text.length; k++) {
    const pc = pieces[k] ?? '';
    const len = pc.length;
    if (len <= 0) continue;
    const take = Math.min(len, text.length - pos);
    if (take <= 0) break;
    segs.push({ text: text.slice(pos, pos + take), special: isSpecialText(pc), idx: k });
    pos += take;
  }
  if (pos < text.length) {
    segs.push({ text: text.slice(pos), special: false, idx: pieces.length });
  }
  return segs;
}

/** One ``/api/v1/tokenize`` call, normalized: ``pieces`` is only kept when
 * it aligns 1:1 with ``ids`` (otherwise highlighting would lie). */
export async function requestTokenize(
  backend: string,
  model: string,
  text: string,
  signal?: AbortSignal
): Promise<TokenizePreview> {
  const data = await apiFetch<TokenizePreview>('/api/v1/tokenize', {
    method: 'POST',
    body: JSON.stringify({ backend, model, text }),
    signal
  });
  return {
    ids: data.ids,
    pieces: data.pieces && data.pieces.length === data.ids.length ? data.pieces : []
  };
}

/** Human-readable tokenize failure for the small status line. */
export function tokenizeErrorText(err: unknown): string {
  if (err instanceof ApiError) return `tokenize failed: HTTP ${err.status}`;
  const msg = err instanceof Error ? err.message : '';
  return `tokenize failed: ${msg || 'unknown'}`;
}
