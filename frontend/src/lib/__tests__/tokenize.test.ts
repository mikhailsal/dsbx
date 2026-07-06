/**
 * Segmentation rules shared by the editable TokenComposer backdrop and the
 * read-only TokenizedText view (chat mode). The invariant that matters:
 * concatenated segment texts ALWAYS equal the source string, so highlight
 * overlays stay aligned with what the user actually sees.
 */

import { describe, expect, it } from 'vitest';
import { segmentPieces } from '../tokenize';

describe('segmentPieces', () => {
  it('slices the source string by piece lengths', () => {
    const segs = segmentPieces('Hello world', ['Hello', ' world']);
    expect(segs.map((s) => s.text)).toEqual(['Hello', ' world']);
    expect(segs.map((s) => s.special)).toEqual([false, false]);
  });

  it('marks special-token pieces', () => {
    const segs = segmentPieces('<|im_start|>user', ['<|im_start|>', 'user']);
    expect(segs[0].special).toBe(true);
    expect(segs[1].special).toBe(false);
  });

  it('covers the whole source even when pieces run short (decode divergence)', () => {
    const segs = segmentPieces('abcdef', ['ab', 'cd']);
    expect(segs.map((s) => s.text).join('')).toBe('abcdef');
    expect(segs[segs.length - 1]).toMatchObject({ text: 'ef', special: false });
  });

  it('never overruns the source when pieces are longer than the text', () => {
    const segs = segmentPieces('ab', ['abcd']);
    expect(segs.map((s) => s.text).join('')).toBe('ab');
  });

  it('returns nothing without pieces (plain-text fallback)', () => {
    expect(segmentPieces('text', [])).toEqual([]);
  });

  it('skips empty pieces without desyncing', () => {
    const segs = segmentPieces('ab', ['a', '', 'b']);
    expect(segs.map((s) => s.text).join('')).toBe('ab');
  });
});
