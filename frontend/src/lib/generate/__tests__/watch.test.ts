/**
 * Watch-column reconstruction tests: header ordering (text before id
 * before EOS, mirroring the server's resolution order), deduplication,
 * and per-row lookup semantics.
 */

import { describe, expect, it } from 'vitest';
import { buildWatchColumns, watchedAt } from '../watch';
import type { StepResult } from '$lib/types';

describe('buildWatchColumns', () => {
  it('orders text, then ids (with cached piece labels), then EOS, deduped', () => {
    const cols = buildWatchColumns({
      watchTexts: [' Paris'],
      watchIds: ['7', '7', 'junk', '9'],
      watchEos: true,
      eosTokenIds: [9, 100],
      tokenCache: { 7: ' the' }
    });
    expect(cols.map((c) => [c.source, c.tokenId])).toEqual([
      ['text', -1],
      ['id', 7],
      ['id', 9],
      ['eos', 100]
    ]);
    expect(cols[1].label).toBe('id=7 " the"');
    expect(cols[3].label).toBe('EOS:100');
  });
});

describe('watchedAt', () => {
  const cand = (id: number) => ({
    token_id: id,
    text: `t${id}`,
    logprob: -1,
    rank: 0,
    is_special: false
  });
  const step: StepResult = {
    position: 0,
    candidates: [],
    is_full_vocab: true,
    chosen: null,
    context_text: null,
    watched: [
      { token_id: 5, candidate: cand(5) },
      { token_id: 7, candidate: cand(7) }
    ]
  };

  it('resolves text columns positionally and id columns by lookup', () => {
    const textCol = { label: 'text:" x"', tokenId: -1, source: 'text' as const };
    const idCol = { label: 'id=7', tokenId: 7, source: 'id' as const };
    expect(watchedAt(step, textCol, 0)?.token_id).toBe(5);
    expect(watchedAt(step, idCol, 0)?.token_id).toBe(7);
    expect(watchedAt(step, { ...idCol, tokenId: 404 }, 0)).toBeNull();
  });
});
