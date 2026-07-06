/**
 * DeepSeek end-to-end engine tests -- the family whose USER turns have no
 * closing marker at all: `<｜User｜>text<｜Assistant｜>` relies on the next
 * turn's opener as the delimiter. The suffix-search parser used to consume
 * everything to the end of the string for such roles, which failed the
 * profile self-test and locked the Raw tab into read-only mode.
 */

import { describe, expect, it } from 'vitest';
import { deriveProfile, prefillMarkers } from '../profile';
import { parseRaw } from '../parse';
import { renderChat } from '../render';
import type { TemplateInputs } from '../types';
import fixtures from '../fixtures/templates.json';

const BOS = '<\uff5cbegin\u2581of\u2581sentence\uff5c>';
const EOS = '<\uff5cend\u2581of\u2581sentence\uff5c>';
const USER = '<\uff5cUser\uff5c>';
const ASST = '<\uff5cAssistant\uff5c>';

const inputs: TemplateInputs = {
  template: fixtures.deepseek.template,
  bosToken: fixtures.deepseek.bos_token,
  eosToken: fixtures.deepseek.eos_token
};
const profile = deriveProfile(inputs);

describe('deepseek profile derivation', () => {
  it('passes the self-test despite the empty user suffix', () => {
    expect(profile.family).toBe('deepseek');
    expect(profile.complete).toBe(true);
    expect(profile.roles.user).toEqual({ prefix: USER, suffix: '' });
    expect(profile.roles.assistant).toEqual({
      prefix: `${ASST}</think>`,
      suffix: EOS
    });
    expect(profile.generationPrompt).toBe(`${ASST}</think>`);
    expect(profile.reasoning).toEqual({ open: '<think>', close: '</think>' });
  });

  it('supports the prefix/scaffold split for prefill rendering', () => {
    expect(prefillMarkers(profile)).toEqual({ prefix: `${ASST}</think>`, scaffold: '' });
  });
});

describe('deepseek raw -> blocks', () => {
  it('parses a closed turn + follow-up user + bare generation prompt', () => {
    // The exact wire shape from the live bug report: append a finished
    // answer, add the next user turn, cue the model again.
    const raw =
      `${BOS}${USER}What is the capital of France?` +
      `${ASST}</think>The capital of France is Paris.${EOS}` +
      `${USER}Are you sure?${ASST}</think>`;
    const res = parseRaw(raw, profile);
    expect(res.ok).toBe(true);
    expect(res.warnings).toEqual([]);
    expect(res.doc.blocks).toEqual([
      { kind: 'user', content: 'What is the capital of France?' },
      { kind: 'assistant', content: 'The capital of France is Paris.' },
      { kind: 'user', content: 'Are you sure?' }
    ]);
    expect(res.doc.addGenerationPrompt).toBe(true);
  });

  it('treats a trailing user turn as CLOSED (no marker needed), not open', () => {
    const raw = `${BOS}${USER}Hello there`;
    const res = parseRaw(raw, profile);
    expect(res.ok).toBe(true);
    expect(res.warnings).toEqual([]);
    expect(res.doc.blocks).toEqual([{ kind: 'user', content: 'Hello there' }]);
    expect(res.doc.addGenerationPrompt).toBe(false);
  });

  it('round-trips an open (prefill) assistant turn byte-for-byte', () => {
    const raw =
      `${BOS}${USER}What is the capital of France?` +
      `${ASST}</think>The capital of France`;
    const res = parseRaw(raw, profile);
    expect(res.ok).toBe(true);
    const last = res.doc.blocks[res.doc.blocks.length - 1];
    expect(last).toEqual({
      kind: 'assistant',
      content: 'The capital of France',
      prefill: true
    });
    const rendered = renderChat(res.doc, inputs, prefillMarkers(profile)?.prefix ?? null);
    expect(rendered.raw).toBe(raw);
  });

  it('recovers the markerless system prompt after the BOS', () => {
    const raw = `${BOS}You are terse.${USER}Hi${ASST}</think>Hello.${EOS}`;
    const res = parseRaw(raw, profile);
    expect(res.ok).toBe(true);
    expect(res.doc.blocks[0]).toEqual({ kind: 'system', content: 'You are terse.' });
    expect(res.doc.blocks[1]).toEqual({ kind: 'user', content: 'Hi' });
  });
});
