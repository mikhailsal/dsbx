/**
 * Reverse-path tests: raw string -> blocks, including the full
 * blocks -> raw -> blocks round trip on real templates and the
 * positioned-error contract for malformed input.
 */

import { describe, expect, it } from 'vitest';
import { parseRaw } from '../parse';
import { deriveProfile } from '../profile';
import { renderChat } from '../render';
import type { ChatBlock, ChatDoc, TemplateInputs } from '../types';
import fixtures from '../fixtures/templates.json';

type FixtureKey = keyof typeof fixtures;

function inputsOf(key: FixtureKey): TemplateInputs {
  const fx = fixtures[key];
  return { template: fx.template, bosToken: fx.bos_token, eosToken: fx.eos_token };
}

const chatmlProfile = deriveProfile(inputsOf('chatml'));

function contentBlocks(blocks: ChatBlock[]): [string, string][] {
  return blocks
    .filter((b): b is Extract<ChatBlock, { content: string }> => 'content' in b)
    .map((b) => [b.kind, b.content.trim()]);
}

describe('round trip: blocks -> raw -> blocks on real templates', () => {
  const doc: ChatDoc = {
    blocks: [
      { kind: 'user', content: 'What is 2+2?' },
      { kind: 'assistant', content: '4' },
      { kind: 'user', content: 'And 3+3?' }
    ],
    addGenerationPrompt: true
  };

  it.each(Object.keys(fixtures) as FixtureKey[])('%s round-trips', (key) => {
    const inputs = inputsOf(key);
    const profile = deriveProfile(inputs);
    const rendered = renderChat(doc, inputs);
    expect(rendered.error).toBeNull();
    const parsed = parseRaw(rendered.raw, profile);
    expect(parsed.error).toBeNull();
    const roles = contentBlocks(parsed.doc.blocks).filter(([, c]) => c.length > 0);
    expect(roles).toEqual([
      ['user', 'What is 2+2?'],
      ['assistant', '4'],
      ['user', 'And 3+3?']
    ]);
  });
});

describe('parseRaw structure handling', () => {
  it('parses a well-formed ChatML conversation with system prompt', () => {
    const raw =
      '<|im_start|>system\nBe terse.<|im_end|>\n' +
      '<|im_start|>user\nhi<|im_end|>\n' +
      '<|im_start|>assistant\n';
    const result = parseRaw(raw, chatmlProfile);
    expect(result.ok).toBe(true);
    expect(contentBlocks(result.doc.blocks)).toEqual([
      ['system', 'Be terse.'],
      ['user', 'hi']
    ]);
    expect(result.doc.addGenerationPrompt).toBe(true);
  });

  it('keeps text after the generation prompt as an open assistant turn (prefill)', () => {
    const raw = '<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nSure, ';
    const result = parseRaw(raw, chatmlProfile);
    expect(result.ok).toBe(true);
    expect(result.doc.addGenerationPrompt).toBe(false);
    const last = result.doc.blocks[result.doc.blocks.length - 1];
    // ``prefill: true`` so blocks -> raw re-renders the SAME open turn
    // instead of appending end-of-turn markers (lossless round trip).
    expect(last).toEqual({ kind: 'assistant', content: 'Sure, ', prefill: true });
    expect(result.warnings.join(' ')).toContain('not closed');
  });

  it('keeps an unterminated turn with an OPEN <think> verbatim as ONE prefill block', () => {
    // The open turn must NOT be split into a reasoning sub-block:
    // sub-blocks re-render through the template (which reworks <think>
    // sections), while the literal text round-trips exactly.
    const raw = '<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n<think>\nHmm, let me';
    const result = parseRaw(raw, chatmlProfile);
    expect(result.ok).toBe(true);
    const last = result.doc.blocks[result.doc.blocks.length - 1];
    expect(last).toEqual({ kind: 'assistant', content: '<think>\nHmm, let me', prefill: true });
    expect(result.doc.blocks.some((b) => b.kind === 'assistant_reasoning')).toBe(false);
    expect(result.warnings.join(' ')).toContain('not closed');
  });

  it('returns a positioned error on a corrupted role marker, keeping prior blocks', () => {
    const good = '<|im_start|>user\nhi<|im_end|>\n';
    const raw = good + '<|im_TYPO|>assistant\nok<|im_end|>\n';
    const result = parseRaw(raw, chatmlProfile);
    expect(result.ok).toBe(false);
    expect(result.error?.position).toBe(good.length);
    expect(result.error?.expected).toContain('<|im_start|>user');
    expect(result.error?.found).toContain('<|im_TYPO|>');
    expect(result.error?.hint).toContain('role marker');
    // The good part survives for the UI to show.
    expect(contentBlocks(result.doc.blocks)).toEqual([['user', 'hi']]);
  });

  it('extracts a reasoning section into an assistant_reasoning block', () => {
    const raw =
      '<|im_start|>user\nq<|im_end|>\n' +
      '<|im_start|>assistant\n<think>\nstep by step\n</think>\n\nanswer<|im_end|>\n';
    const result = parseRaw(raw, chatmlProfile);
    expect(result.ok).toBe(true);
    expect(contentBlocks(result.doc.blocks)).toEqual([
      ['user', 'q'],
      ['assistant_reasoning', 'step by step'],
      ['assistant', 'answer']
    ]);
  });

  it('flags an unclosed reasoning section but keeps its text', () => {
    const raw = '<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n<think>hmm<|im_end|>\n';
    const result = parseRaw(raw, chatmlProfile);
    expect(result.ok).toBe(true);
    expect(result.warnings.join(' ')).toContain('missing its closing');
    expect(
      result.doc.blocks.some((b) => b.kind === 'assistant_reasoning' && b.content.includes('hmm'))
    ).toBe(true);
  });

  it('extracts Hermes-style tool calls into tool_call blocks', () => {
    const raw =
      '<|im_start|>user\nweather?<|im_end|>\n' +
      '<|im_start|>assistant\n<tool_call>\n' +
      '{"name": "get_weather", "arguments": {"city": "Paris"}}' +
      '\n</tool_call><|im_end|>\n';
    const result = parseRaw(raw, chatmlProfile);
    expect(result.ok).toBe(true);
    const call = result.doc.blocks.find((b) => b.kind === 'tool_call');
    expect(call).toEqual({
      kind: 'tool_call',
      name: 'get_weather',
      argumentsJson: '{"city":"Paris"}'
    });
  });

  it('degrades malformed tool-call JSON to assistant text with a warning', () => {
    const raw =
      '<|im_start|>user\nq<|im_end|>\n' +
      '<|im_start|>assistant\n<tool_call>\nnot json\n</tool_call><|im_end|>\n';
    const result = parseRaw(raw, chatmlProfile);
    expect(result.ok).toBe(true);
    expect(result.warnings.join(' ')).toContain('not valid JSON');
    expect(result.doc.blocks.some((b) => b.kind === 'tool_call')).toBe(false);
  });

  it('parses tool_response turns into tool_result blocks (Qwen convention)', () => {
    const inputs = inputsOf('qwen3');
    const profile = deriveProfile(inputs);
    const raw =
      '<|im_start|>user\nweather?<|im_end|>\n' +
      '<|im_start|>user\n<tool_response>\n18C\n</tool_response><|im_end|>\n';
    const result = parseRaw(raw, profile);
    expect(result.ok).toBe(true);
    expect(contentBlocks(result.doc.blocks)).toEqual([
      ['user', 'weather?'],
      ['tool_result', '18C']
    ]);
  });

  it('refuses to guess for an incomplete profile, keeping the raw text', () => {
    const profile = deriveProfile({ template: null, bosToken: null, eosToken: null });
    const result = parseRaw('some raw prompt', profile);
    expect(result.ok).toBe(false);
    expect(result.error?.hint).toContain('Raw mode');
    expect(result.doc.blocks).toEqual([{ kind: 'assistant', content: 'some raw prompt' }]);
  });
});
