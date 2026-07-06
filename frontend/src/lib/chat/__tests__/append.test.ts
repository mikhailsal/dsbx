/**
 * "Append as assistant" doc-transform tests, plus the full open-turn
 * loop (derive profile -> append -> render -> parse) on a template that
 * injects a reasoning scaffold into its generation prompt (the Qwen3.5
 * shape that motivated the scaffold handling).
 */

import { describe, expect, it } from 'vitest';
import { appendAssistantText } from '../append';
import { deriveProfile, prefillMarkers } from '../profile';
import { parseRaw } from '../parse';
import { renderChat } from '../render';
import type { ChatDoc, TemplateInputs } from '../types';

const baseDoc: ChatDoc = {
  blocks: [{ kind: 'user', content: 'q' }],
  addGenerationPrompt: true
};

// A minimal ChatML template whose generation prompt carries a "<think>\n"
// scaffold beyond the assistant turn opener -- the Qwen3.5 behaviour.
const SCAFFOLD_TEMPLATE =
  '{%- for message in messages %}' +
  "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}" +
  '{%- endfor %}' +
  "{%- if add_generation_prompt %}{{ '<|im_start|>assistant\\n<think>\\n' }}{%- endif %}";
const scaffoldInputs: TemplateInputs = {
  template: SCAFFOLD_TEMPLATE,
  bosToken: null,
  eosToken: '<|im_end|>'
};
const scaffoldProfile = deriveProfile(scaffoldInputs);

describe('appendAssistantText', () => {
  it('creates an open prefill block, folding in the generation-prompt scaffold', () => {
    const doc = appendAssistantText(baseDoc, 'Hmm, let me', false, scaffoldProfile);
    expect(doc.blocks[1]).toEqual({
      kind: 'assistant',
      content: '<think>\nHmm, let me',
      prefill: true
    });
    expect(doc.addGenerationPrompt).toBe(false);
  });

  it('merges consecutive appends into the SAME open turn (no block splitting)', () => {
    let doc = appendAssistantText(baseDoc, 'part one', false, scaffoldProfile);
    doc = appendAssistantText(doc, ' part two', false, scaffoldProfile);
    expect(doc.blocks).toHaveLength(2);
    expect(doc.blocks[1]).toEqual({
      kind: 'assistant',
      content: '<think>\npart one part two',
      prefill: true
    });
  });

  it('closes the merged turn when the continuation finished, splitting reasoning out', () => {
    // FINISHED turns are re-rendered by the template itself (Qwen wraps
    // reasoning_content in its own <think> scaffold), so the literal tags
    // must become an assistant_reasoning sub-block -- keeping them in the
    // content would double the scaffold in the raw prompt.
    let doc = appendAssistantText(baseDoc, 'plan', false, scaffoldProfile);
    doc = appendAssistantText(doc, ' done</think>Paris.<|im_end|>', true, scaffoldProfile, '<|im_end|>');
    expect(doc.blocks.slice(1)).toEqual([
      { kind: 'assistant_reasoning', content: '\nplan done' },
      { kind: 'assistant', content: 'Paris.' }
    ]);
  });

  it('keeps a truncated turn verbatim, open <think> tag included', () => {
    const doc = appendAssistantText(baseDoc, 'thinking still…', false, scaffoldProfile);
    expect(doc.blocks[1]).toEqual({
      kind: 'assistant',
      content: '<think>\nthinking still…',
      prefill: true
    });
  });

  it('adds no scaffold when the doc did not cue the model', () => {
    const doc = appendAssistantText(
      { ...baseDoc, addGenerationPrompt: false },
      'x',
      false,
      scaffoldProfile
    );
    expect(doc.blocks[1]).toEqual({ kind: 'assistant', content: 'x', prefill: true });
  });

  it('handles a null profile: plain closed block, no scaffold, eos stripped', () => {
    const doc = appendAssistantText(baseDoc, 'plain answer</s>', true, null, '</s>');
    expect(doc.blocks[1]).toEqual({ kind: 'assistant', content: 'plain answer' });
  });

  it('returns the doc unchanged for empty text', () => {
    expect(appendAssistantText(baseDoc, '', false, scaffoldProfile)).toBe(baseDoc);
  });
});

describe('open-turn loop on a scaffolded generation prompt', () => {
  const markers = prefillMarkers(scaffoldProfile);

  it('derives the assistant prefix and the scaffold separately', () => {
    expect(scaffoldProfile.complete).toBe(true);
    expect(markers).toEqual({ prefix: '<|im_start|>assistant\n', scaffold: '<think>\n' });
  });

  it('append -> render reproduces the exact mid-turn context, scaffold visible once', () => {
    const doc = appendAssistantText(baseDoc, 'Hmm, let me', false, scaffoldProfile);
    const rendered = renderChat(doc, scaffoldInputs, markers?.prefix ?? null);
    expect(rendered.error).toBeNull();
    expect(rendered.raw).toBe(
      '<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n<think>\nHmm, let me'
    );
  });

  it('render -> parse round-trips the open turn verbatim (tags in the block)', () => {
    const doc = appendAssistantText(baseDoc, 'Hmm, let me', false, scaffoldProfile);
    const rendered = renderChat(doc, scaffoldInputs, markers?.prefix ?? null);
    const parsed = parseRaw(rendered.raw, scaffoldProfile);
    expect(parsed.ok).toBe(true);
    expect(parsed.doc.blocks).toEqual([
      { kind: 'user', content: 'q' },
      { kind: 'assistant', content: '<think>\nHmm, let me', prefill: true }
    ]);
    // And re-rendering the parsed doc is byte-identical (lossless loop).
    const again = renderChat(parsed.doc, scaffoldInputs, markers?.prefix ?? null);
    expect(again.raw).toBe(rendered.raw);
  });

  it('a bare generation prompt still parses as add_generation_prompt, not prefill', () => {
    const rendered = renderChat(baseDoc, scaffoldInputs, markers?.prefix ?? null);
    const parsed = parseRaw(rendered.raw, scaffoldProfile);
    expect(parsed.ok).toBe(true);
    expect(parsed.doc.addGenerationPrompt).toBe(true);
    expect(parsed.doc.blocks).toEqual([{ kind: 'user', content: 'q' }]);
  });
});
