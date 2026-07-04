/**
 * Snippet-generation tests: the raw-mode smart buttons must be
 * model-correct by construction (built from the derived profile).
 */

import { describe, expect, it } from 'vitest';
import { deriveProfile } from '../profile';
import { buildSnippets } from '../snippets';
import type { TemplateInputs } from '../types';
import fixtures from '../fixtures/templates.json';

function inputsOf(key: keyof typeof fixtures): TemplateInputs {
  const fx = fixtures[key];
  return { template: fx.template, bosToken: fx.bos_token, eosToken: fx.eos_token };
}

describe('buildSnippets', () => {
  it('produces ChatML-correct turn buttons for a ChatML model', () => {
    const inputs = inputsOf('chatml');
    const snippets = buildSnippets(deriveProfile(inputs), inputs);
    const byId = Object.fromEntries(snippets.map((s) => [s.id, s]));

    expect(byId.user.text).toBe('<|im_start|>user\n<|im_end|>\n');
    expect(byId.user.cursorBack).toBe('<|im_end|>\n'.length);
    expect(byId.assistant.text).toBe('<|im_start|>assistant\n<|im_end|>\n');
    expect(byId.system.text).toBe('<|im_start|>system\n<|im_end|>\n');
    expect(byId['generation-prompt'].text).toBe('<|im_start|>assistant\n');
    expect(byId.thinking.text).toBe('<think></think>');
    expect(byId.eos.text).toBe('<|im_end|>');
  });

  it('produces Gemma-correct buttons: model role, no system, no thinking', () => {
    const inputs = inputsOf('gemma2');
    const snippets = buildSnippets(deriveProfile(inputs), inputs);
    const ids = snippets.map((s) => s.id);

    expect(ids).not.toContain('system');
    expect(ids).not.toContain('thinking');
    expect(snippets.find((s) => s.id === 'assistant')?.text).toBe(
      '<start_of_turn>model\n<end_of_turn>\n'
    );
    expect(snippets.find((s) => s.id === 'preamble')?.text).toBe('<bos>');
  });

  it('places the tool-call cursor inside the name field', () => {
    const inputs = inputsOf('chatml');
    const snippets = buildSnippets(deriveProfile(inputs), inputs);
    const tc = snippets.find((s) => s.id === 'tool-call');
    expect(tc).toBeDefined();
    const insertAt = tc!.text.length - tc!.cursorBack;
    expect(tc!.text.slice(0, insertAt)).toMatch(/"name": "$/);
  });

  it('emits no turn buttons for an incomplete profile beyond what was derived', () => {
    const snippets = buildSnippets(
      deriveProfile({ template: null, bosToken: null, eosToken: '</s>' }),
      { template: null, bosToken: null, eosToken: '</s>' }
    );
    // Only the EOS button survives -- no derived markers to build from.
    expect(snippets.map((s) => s.id)).toEqual(['eos']);
  });
});
