/**
 * Forward-path tests: blocks -> messages -> raw string.
 *
 * The fixture file carries REAL chat templates fetched from the HF Hub
 * plus ground-truth renders produced by Python jinja2 configured exactly
 * like transformers' chat-template environment -- so these tests pin the
 * browser engine (@huggingface/jinja) to what the model server actually
 * does.
 */

import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import { FALLBACK_CHATML_TEMPLATE, blocksToMessages, renderChat } from '../render';
import type { ChatBlock, ChatDoc, TemplateInputs } from '../types';
import fixtures from '../fixtures/templates.json';

type FixtureKey = keyof typeof fixtures;

// The jinja2 ground truth was generated with strftime_now pinned to this
// date (gpt-oss stamps "Current date" into its system header); the jinja
// engine reads the real clock, so fake it for the fixture comparisons.
beforeAll(() => {
  vi.useFakeTimers();
  vi.setSystemTime(new Date(2026, 0, 15, 12, 0, 0));
});
afterAll(() => {
  vi.useRealTimers();
});

function inputsOf(key: FixtureKey): TemplateInputs {
  const fx = fixtures[key];
  return { template: fx.template, bosToken: fx.bos_token, eosToken: fx.eos_token };
}

function docOf(key: FixtureKey): ChatDoc {
  const conv = fixtures[key].expected.conversation as { role: string; content: string }[];
  const blocks = conv.map(
    (m) => ({ kind: m.role, content: m.content }) as ChatBlock
  );
  return { blocks, addGenerationPrompt: fixtures[key].expected.add_generation_prompt };
}

describe('renderChat against transformers ground truth', () => {
  for (const key of Object.keys(fixtures) as FixtureKey[]) {
    it(`renders the ${key} fixture identically to Python jinja2`, () => {
      const result = renderChat(docOf(key), inputsOf(key));
      expect(result.error).toBeNull();
      expect(result.raw).toBe(fixtures[key].expected.rendered);
      expect(result.usedFallback).toBe(false);
    });
  }

  it('reports template raise_exception verbatim instead of throwing (gemma + system)', () => {
    const doc: ChatDoc = {
      blocks: [
        { kind: 'system', content: 'You are terse.' },
        { kind: 'user', content: 'hi' }
      ],
      addGenerationPrompt: true
    };
    const result = renderChat(doc, inputsOf('gemma2'));
    expect(result.raw).toBe('');
    expect(result.error).toContain('System role not supported');
  });

  it('falls back to ChatML when the model ships no template', () => {
    const doc: ChatDoc = {
      blocks: [{ kind: 'user', content: 'hi' }],
      addGenerationPrompt: true
    };
    const result = renderChat(doc, { template: null, bosToken: null, eosToken: null });
    expect(result.usedFallback).toBe(true);
    expect(result.raw).toBe('<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n');
    expect(result.warnings.join(' ')).toContain('fallback');
  });

  it('keeps the fallback template in sync with the backend constant shape', () => {
    // The Python twin lives in dsbx/core/chat_template.py; both must produce
    // the same ChatML scaffold. Guard the structural markers.
    expect(FALLBACK_CHATML_TEMPLATE).toContain('<|im_start|>');
    expect(FALLBACK_CHATML_TEMPLATE).toContain('add_generation_prompt');
  });
});

describe('blocksToMessages', () => {
  it('attaches a reasoning block to the following assistant message', () => {
    const { messages, warnings } = blocksToMessages([
      { kind: 'user', content: 'q' },
      { kind: 'assistant_reasoning', content: 'let me think' },
      { kind: 'assistant', content: 'a' }
    ]);
    expect(warnings).toEqual([]);
    expect(messages).toEqual([
      { role: 'user', content: 'q' },
      { role: 'assistant', content: 'a', reasoning_content: 'let me think' }
    ]);
  });

  it('emits a trailing reasoning block as its own assistant message with a warning', () => {
    const { messages, warnings } = blocksToMessages([
      { kind: 'user', content: 'q' },
      { kind: 'assistant_reasoning', content: 'hmm' }
    ]);
    expect(messages[1]).toEqual({ role: 'assistant', content: '', reasoning_content: 'hmm' });
    expect(warnings[0]).toContain('reasoning block');
  });

  it('merges consecutive tool calls into one assistant message', () => {
    const { messages } = blocksToMessages([
      { kind: 'user', content: 'weather in Paris and Rome?' },
      { kind: 'tool_call', name: 'get_weather', argumentsJson: '{"city": "Paris"}' },
      { kind: 'tool_call', name: 'get_weather', argumentsJson: '{"city": "Rome"}' },
      { kind: 'tool_result', content: '18C', name: 'get_weather', callId: 'call_1' }
    ]);
    expect(messages).toHaveLength(3);
    const assistant = messages[1];
    expect(assistant.role).toBe('assistant');
    expect(assistant.content).toBeNull();
    expect(assistant.tool_calls).toHaveLength(2);
    expect(assistant.tool_calls?.[0].function.name).toBe('get_weather');
    expect(messages[2]).toEqual({
      role: 'tool',
      content: '18C',
      name: 'get_weather',
      tool_call_id: 'call_1'
    });
  });

  it('collects tool definitions into tools, not messages', () => {
    const defs = [{ type: 'function', function: { name: 'f', parameters: {} } }];
    const { messages, tools } = blocksToMessages([
      { kind: 'tool_defs', toolsJson: JSON.stringify(defs) },
      { kind: 'user', content: 'q' }
    ]);
    expect(messages).toEqual([{ role: 'user', content: 'q' }]);
    expect(tools).toEqual(defs);
  });

  it('warns on malformed tool definitions instead of dropping the conversation', () => {
    const { messages, tools, warnings } = blocksToMessages([
      { kind: 'tool_defs', toolsJson: 'not json' },
      { kind: 'user', content: 'q' }
    ]);
    expect(messages).toHaveLength(1);
    expect(tools).toBeNull();
    expect(warnings[0]).toContain('tool definitions ignored');
  });
});

