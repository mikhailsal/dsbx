/**
 * Harmony (gpt-oss) end-to-end engine tests -- the family whose template
 * closes the FINAL assistant turn with ``<|return|>`` while mid-stream
 * turns close with ``<|end|>``. That asymmetry used to degenerate the
 * derived assistant suffix to ``<|`` (the lcp of the two closers), which
 * false-matched inside every special token and made raw -> blocks
 * parsing reject any appended completion -- the "format error with no
 * way out" bug.
 */

import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import { appendAssistantText } from '../append';
import { deriveProfile, prefillMarkers } from '../profile';
import { parseRaw } from '../parse';
import { renderChat } from '../render';
import type { ChatDoc, TemplateInputs } from '../types';
import fixtures from '../fixtures/templates.json';

const inputs: TemplateInputs = {
  template: fixtures.gptoss.template,
  bosToken: fixtures.gptoss.bos_token,
  eosToken: fixtures.gptoss.eos_token
};

// gp-oss stamps "Current date" into its system header via strftime_now,
// so BOTH the profile (whose preamble carries the date) and the renders
// must happen under the same frozen clock.
let profile: ReturnType<typeof deriveProfile>;
beforeAll(() => {
  vi.useFakeTimers();
  vi.setSystemTime(new Date(2026, 0, 15, 12, 0, 0));
  profile = deriveProfile(inputs);
});
afterAll(() => {
  vi.useRealTimers();
});

const doc: ChatDoc = {
  blocks: [{ kind: 'user', content: 'How do I make a custom GPT?' }],
  addGenerationPrompt: true
};

describe('harmony profile derivation', () => {
  it('derives whole-token markers (curated <|end|> suffix, not the lcp fragment)', () => {
    expect(profile.complete).toBe(true);
    expect(profile.roles.user).toEqual({
      prefix: '<|start|>user<|message|>',
      suffix: '<|end|>'
    });
    expect(profile.roles.assistant).toEqual({
      prefix: '<|start|>assistant<|channel|>final<|message|>',
      suffix: '<|end|>'
    });
    expect(profile.generationPrompt).toBe('<|start|>assistant');
    expect(profile.lastAssistantSuffix).toBe('<|return|>');
  });

  it('declines the prefix/scaffold split (channel-scoped assistant opener)', () => {
    expect(prefillMarkers(profile)).toBeNull();
  });
});

describe('harmony raw append round trips', () => {
  it('a TRUNCATED analysis completion parses as one open (prefill) turn', () => {
    // Raw-mode flow: render, run, append the streamed text, switch to
    // Blocks. This is the exact sequence that used to throw the parse
    // error and strand the user in raw mode.
    const rendered = renderChat(doc, inputs, null);
    const completion = '<|channel|>analysis<|message|>The user wants a detailed guide on';
    const raw = rendered.raw + completion;
    const parsed = parseRaw(raw, profile);
    expect(parsed.ok).toBe(true);
    expect(parsed.doc.blocks[parsed.doc.blocks.length - 1]).toEqual({
      kind: 'assistant',
      content: completion,
      prefill: true
    });
    // Blocks -> raw reproduces the mid-turn context byte-for-byte.
    const again = renderChat(parsed.doc, inputs, null);
    expect(again.raw).toBe(raw);
  });

  it('a truncated MULTI-turn run (closed analysis + open final) folds into ONE open block', () => {
    // The model finished its analysis turn and was cut off mid-answer.
    // The whole trailing assistant run must stay verbatim in one prefill
    // block: splitting it would re-render the reasoning through the
    // template (which drops/reworks it) and corrupt the context.
    const rendered = renderChat(doc, inputs, null);
    const completion =
      '<|channel|>analysis<|message|>User asks about GPTs.<|end|>' +
      '<|start|>assistant<|channel|>final<|message|>Head to the';
    const raw = rendered.raw + completion;
    const parsed = parseRaw(raw, profile);
    expect(parsed.ok).toBe(true);
    expect(parsed.doc.blocks.slice(1)).toEqual([
      { kind: 'assistant', content: completion, prefill: true }
    ]);
    const again = renderChat(parsed.doc, inputs, null);
    expect(again.raw).toBe(raw);
  });

  it('a <|return|>-closed final turn parses as a CLOSED turn and round-trips', () => {
    const rendered = renderChat(doc, inputs, null);
    const completion =
      '<|channel|>analysis<|message|>User asks about GPTs.<|end|>' +
      '<|start|>assistant<|channel|>final<|message|>Use the builder.<|return|>';
    const raw = rendered.raw + completion;
    const parsed = parseRaw(raw, profile);
    expect(parsed.ok).toBe(true);
    expect(parsed.doc.addGenerationPrompt).toBe(false);
    expect(parsed.doc.blocks.slice(1)).toEqual([
      { kind: 'assistant_reasoning', content: 'User asks about GPTs.' },
      { kind: 'assistant', content: 'Use the builder.' }
    ]);
    // The template consumes the reasoning via its ``thinking`` key and
    // re-renders the same analysis + final channels.
    const again = renderChat(parsed.doc, inputs, null);
    expect(again.error).toBeNull();
    expect(again.raw).toBe(raw);
  });
});

describe('harmony append as assistant', () => {
  it('normalizes a FINISHED run into reasoning + answer blocks', () => {
    const completion =
      '<|channel|>analysis<|message|>User asks about GPTs.<|end|>' +
      '<|start|>assistant<|channel|>final<|message|>Use the builder.<|return|>';
    const doc2 = appendAssistantText(doc, completion, true, profile, inputs.eosToken);
    expect(doc2.blocks.slice(1)).toEqual([
      { kind: 'assistant_reasoning', content: 'User asks about GPTs.' },
      { kind: 'assistant', content: 'Use the builder.' }
    ]);
    const rendered = renderChat(doc2, inputs, null);
    expect(rendered.error).toBeNull();
    expect(
      rendered.raw.endsWith(
        '<|start|>assistant<|channel|>analysis<|message|>User asks about GPTs.<|end|>' +
          '<|start|>assistant<|channel|>final<|message|>Use the builder.<|return|>'
      )
    ).toBe(true);
  });

  it('keeps a TRUNCATED run verbatim as an open prefill turn', () => {
    const completion = '<|channel|>analysis<|message|>The user wants';
    const doc2 = appendAssistantText(doc, completion, false, profile, inputs.eosToken);
    expect(doc2.blocks[1]).toEqual({ kind: 'assistant', content: completion, prefill: true });
    const rendered = renderChat(doc2, inputs, null);
    expect(rendered.raw.endsWith('<|start|>assistant' + completion)).toBe(true);
  });
});
