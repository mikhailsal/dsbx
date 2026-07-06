/**
 * Wire-assembly tests for the generate-stream request builder: sampler
 * param selection, logit-bias row parsing, chip parsing, and the full
 * body shape for both prompt and chat-simulation sources.
 */

import { describe, expect, it } from 'vitest';
import {
  buildGenerateRequest,
  buildSamplerParams,
  collectLogitBias,
  parseIdChips,
  type GenerateRequestConfig,
  type SamplerKnobs
} from '../request';

const knobs: SamplerKnobs = {
  temperature: 0.7,
  topK: 40,
  topP: 0.9,
  minP: 0.05,
  typicalP: 0.95,
  mirostatTarget: 5.0,
  mirostatLr: 0.1,
  repetitionPenalty: 1.0,
  frequencyPenalty: 0.0,
  presencePenalty: 0.0
};

describe('buildSamplerParams', () => {
  it('greedy sends no params when penalties are at their no-op defaults', () => {
    expect(buildSamplerParams('greedy', knobs)).toEqual({});
  });

  it('picks the knobs the selected sampler actually uses', () => {
    expect(buildSamplerParams('top_p', knobs)).toEqual({ temperature: 0.7, top_p: 0.9 });
    expect(buildSamplerParams('mirostat', knobs)).toEqual({
      temperature: 0.7,
      mirostat_target: 5.0,
      mirostat_lr: 0.1
    });
  });

  it('penalties ride along on every sampler only when non-default', () => {
    const withPenalties = { ...knobs, repetitionPenalty: 1.2, presencePenalty: 0.5 };
    expect(buildSamplerParams('greedy', withPenalties)).toEqual({
      repetition_penalty: 1.2,
      presence_penalty: 0.5
    });
  });
});

describe('collectLogitBias', () => {
  const rows = [
    { id: 'a', tokenId: '42', bias: '-100' },
    { id: 'b', tokenId: 'oops', bias: '5' },
    { id: 'c', tokenId: '7', bias: '150' },
    { id: 'd', tokenId: '9', bias: '2.5' }
  ];

  it('drops unparseable ids and out-of-range biases, keys as strings', () => {
    expect(collectLogitBias(rows, true)).toEqual({ '42': -100, '9': 2.5 });
  });

  it('returns undefined (field omitted) when unsupported or empty', () => {
    expect(collectLogitBias(rows, false)).toBeUndefined();
    expect(collectLogitBias([], true)).toBeUndefined();
  });
});

describe('parseIdChips', () => {
  it('parses finite ints and drops garbage', () => {
    expect(parseIdChips(['1', ' 2', 'x', '', '3'])).toEqual([1, 2, 3]);
  });
});

function baseConfig(): GenerateRequestConfig {
  return {
    backend: 'fw',
    model: 'some-model',
    prompt: 'Once upon a time',
    messages: null,
    tools: null,
    sampler: 'greedy',
    samplerKnobs: knobs,
    maxTokens: 20,
    alternatives: 8,
    stopTexts: ['.'],
    stopIds: ['5', 'junk'],
    seed: 0,
    respectEos: true,
    includePrompt: true,
    serviceTier: 'priority',
    serviceTierSupported: false,
    logitBiasRows: [],
    logitBiasSupported: false,
    echoLast: 4,
    combinedEchoStreamSupported: false,
    watchTexts: [' Paris'],
    watchIds: ['11'],
    watchEos: false,
    prependTokenIds: ['1', '2'],
    prependSupported: true
  };
}

describe('buildGenerateRequest', () => {
  it('builds the raw-prompt body and reports the prepend count', () => {
    const { body, prependCount } = buildGenerateRequest(baseConfig(), {});
    expect(prependCount).toBe(2);
    expect(body).toMatchObject({
      backend: 'fw',
      model: 'some-model',
      prompt: 'Once upon a time',
      max_tokens: 20,
      top_k: 8,
      stop_ids: [5],
      watch_ids: [11],
      prefix_token_ids: [],
      prepend_token_ids: [1, 2]
    });
    // Unsupported / chat-only fields are omitted, not nulled.
    expect(body.messages).toBeUndefined();
    expect(body.service_tier).toBeUndefined();
    expect(body.logit_bias).toBeUndefined();
    expect(body.echo_last).toBeUndefined();
    expect(body.session_id).toBeUndefined();
  });

  it('empty model becomes undefined so the server picks its default', () => {
    const { body } = buildGenerateRequest({ ...baseConfig(), model: '' }, {});
    expect(body.model).toBeUndefined();
  });

  it('gates prepend ids on capability', () => {
    const { body, prependCount } = buildGenerateRequest(
      { ...baseConfig(), prependSupported: false },
      {}
    );
    expect(prependCount).toBe(0);
    expect(body.prepend_token_ids).toEqual([]);
  });

  it('echo_last only ships when include_prompt + combined echo are on', () => {
    const on = buildGenerateRequest(
      { ...baseConfig(), combinedEchoStreamSupported: true },
      {}
    );
    expect(on.body.echo_last).toBe(4);
    const off = buildGenerateRequest(
      { ...baseConfig(), combinedEchoStreamSupported: true },
      { includePromptOverride: false }
    );
    expect(off.body.echo_last).toBeUndefined();
  });

  it('manual mode pins session UUIDs and the picked prefix', () => {
    const { body } = buildGenerateRequest(baseConfig(), {
      maxTokensOverride: 1,
      prefix: [10, 20],
      forManual: true,
      manualSessionId: 'sess',
      manualCacheKey: 'cache'
    });
    expect(body.max_tokens).toBe(1);
    expect(body.prefix_token_ids).toEqual([10, 20]);
    expect(body.session_id).toBe('sess');
    expect(body.prompt_cache_key).toBe('cache');
  });

  it('chat simulation ships messages[] and tools alongside the preview prompt', () => {
    const messages = [{ role: 'user', content: 'hi' }];
    const tools = [{ type: 'function' }];
    const { body } = buildGenerateRequest(
      { ...baseConfig(), messages, tools, prompt: '<preview>' },
      {}
    );
    expect(body.messages).toEqual(messages);
    expect(body.tools).toEqual(tools);
    expect(body.prompt).toBe('<preview>');
  });

  it('omits empty tools arrays', () => {
    const { body } = buildGenerateRequest(
      { ...baseConfig(), messages: [{ role: 'user', content: 'hi' }], tools: [] },
      {}
    );
    expect(body.tools).toBeUndefined();
  });
});
