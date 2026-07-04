/**
 * Profile-derivation tests: real templates in, marker maps out.
 *
 * The strongest assertion here is ``complete: true`` -- it means the
 * derived markers survived the built-in round-trip self-test (render a
 * sentinel conversation with the REAL template, parse it back, compare).
 * Per-family marker checks then pin the interesting specifics.
 */

import { describe, expect, it } from 'vitest';
import { deriveProfile, detectFamily } from '../profile';
import type { TemplateInputs } from '../types';
import fixtures from '../fixtures/templates.json';

type FixtureKey = keyof typeof fixtures;

function inputsOf(key: FixtureKey): TemplateInputs {
  const fx = fixtures[key];
  return { template: fx.template, bosToken: fx.bos_token, eosToken: fx.eos_token };
}

describe('deriveProfile on real templates', () => {
  it.each(Object.keys(fixtures) as FixtureKey[])(
    '%s passes the round-trip self-test',
    (key) => {
      const profile = deriveProfile(inputsOf(key));
      expect(profile.complete).toBe(true);
      expect(profile.roles.user).toBeDefined();
      expect(profile.roles.assistant).toBeDefined();
    }
  );

  it('recovers the ChatML markers exactly', () => {
    const profile = deriveProfile(inputsOf('chatml'));
    expect(profile.family).toBe('chatml');
    expect(profile.roles.user).toEqual({
      prefix: '<|im_start|>user\n',
      suffix: '<|im_end|>\n'
    });
    expect(profile.roles.assistant).toEqual({
      prefix: '<|im_start|>assistant\n',
      suffix: '<|im_end|>\n'
    });
    expect(profile.roles.system?.prefix).toBe('<|im_start|>system\n');
    expect(profile.generationPrompt).toBe('<|im_start|>assistant\n');
    expect(profile.systemSupported).toBe(true);
    // Qwen2.5 injects a default system prompt when the conversation has
    // none -- it lands in the preamble, which the parser treats as
    // optional.
    expect(profile.preamble).toContain('You are Qwen');
  });

  it('recovers Llama-3 markers including the always-appended assistant header', () => {
    const profile = deriveProfile(inputsOf('llama31'));
    expect(profile.family).toBe('llama3');
    expect(profile.preamble).toContain('<|begin_of_text|>');
    // This NousResearch variant appends the assistant header
    // unconditionally, so it surfaces in the user suffix and the derived
    // generation prompt is empty -- the self-test proves that still
    // parses correctly.
    expect(profile.roles.user?.suffix).toContain('<|eot_id|>');
    expect(profile.generationPrompt).toBe('');
  });

  it('maps Gemma assistant turns to the "model" role marker and flags no-system', () => {
    const profile = deriveProfile(inputsOf('gemma2'));
    expect(profile.family).toBe('gemma');
    expect(profile.roles.assistant?.prefix).toBe('<start_of_turn>model\n');
    expect(profile.systemSupported).toBe(false);
    expect(profile.notes.join(' ')).toContain('rejects system messages');
  });

  it('detects Mistral-style system merging', () => {
    const profile = deriveProfile(inputsOf('mistral'));
    expect(profile.family).toBe('mistral');
    expect(profile.roles.user).toEqual({ prefix: '[INST] ', suffix: '[/INST]' });
    // v0.3 raises on system-first sentinel probing, so it reports as
    // unsupported rather than merged -- either way there is no standalone
    // system marker, which is the fact the UI needs.
    expect(profile.roles.system).toBeUndefined();
  });

  it('keeps the Qwen3 assistant prefix clean of the last-turn <think> scaffold', () => {
    const profile = deriveProfile(inputsOf('qwen3'));
    // Qwen3 injects an empty <think> scaffold only into the LAST assistant
    // turn; markers are derived mid-conversation so it must not leak in.
    expect(profile.roles.assistant?.prefix).toBe('<|im_start|>assistant\n');
    expect(profile.reasoning).toEqual({ open: '<think>', close: '</think>' });
    expect(profile.roles.tool?.prefix).toContain('<tool_response>');
    expect(profile.roles.tool?.suffix).not.toContain('<think>');
  });

  it('degrades to raw-only for a null template', () => {
    const profile = deriveProfile({ template: null, bosToken: null, eosToken: null });
    expect(profile.complete).toBe(false);
    expect(profile.notes.join(' ')).toContain('no chat template');
  });

  it('degrades gracefully on a template that always raises', () => {
    const profile = deriveProfile({
      template: "{{ raise_exception('nope') }}",
      bosToken: null,
      eosToken: null
    });
    expect(profile.complete).toBe(false);
    expect(profile.notes.join(' ')).toContain('marker derivation failed');
  });
});

describe('detectFamily', () => {
  it('fingerprints each known family', () => {
    expect(detectFamily(fixtures.llama31.template)).toBe('llama3');
    expect(detectFamily(fixtures.gptoss.template)).toBe('harmony');
    expect(detectFamily(fixtures.gemma2.template)).toBe('gemma');
    expect(detectFamily(fixtures.mistral.template)).toBe('mistral');
    expect(detectFamily(fixtures.chatml.template)).toBe('chatml');
    expect(detectFamily('{{ messages }}')).toBe('unknown');
  });
});
