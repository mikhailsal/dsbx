/**
 * Frame-dispatch tests for the generate-stream consumer: each SSE frame
 * kind routes to its handler with normalized payloads, and unknown
 * frames are ignored (forward compatibility with newer servers).
 */

import { describe, expect, it, vi } from 'vitest';
import { consumeGenerateStream, dispatchGenerateEvent } from '../stream';
import type { GenerateStreamHandlers } from '../stream';
import type { SseFrame } from '$lib/sse';

function handlers(): GenerateStreamHandlers {
  return {
    onStep: vi.fn(),
    onPromptScore: vi.fn(),
    onPerf: vi.fn(),
    onRawOutput: vi.fn(),
    onUsage: vi.fn(),
    onDone: vi.fn()
  };
}

describe('dispatchGenerateEvent', () => {
  it('routes step frames', () => {
    const h = handlers();
    const step = { step: 0, decision: { token_id: 1, token_text: 'a' } };
    dispatchGenerateEvent({ event: 'step', step }, h);
    expect(h.onStep).toHaveBeenCalledWith(step);
  });

  it('routes prompt_score with defaults for missing fields', () => {
    const h = handlers();
    dispatchGenerateEvent({ event: 'prompt_score' }, h);
    expect(h.onPromptScore).toHaveBeenCalledWith([], '');
  });

  it('normalizes usage counters to explicit nulls', () => {
    const h = handlers();
    dispatchGenerateEvent({ event: 'usage', requests: 2, prompt_tokens: 5 }, h);
    expect(h.onUsage).toHaveBeenCalledWith({
      requests: 2,
      prompt_tokens: 5,
      completion_tokens: null,
      total_tokens: null,
      notes: []
    });
  });

  it('perf / raw_output degrade to null on malformed payloads', () => {
    const h = handlers();
    dispatchGenerateEvent({ event: 'perf', metrics: 'nope' } as unknown as SseFrame, h);
    dispatchGenerateEvent({ event: 'raw_output' }, h);
    expect(h.onPerf).toHaveBeenCalledWith(null);
    expect(h.onRawOutput).toHaveBeenCalledWith(null);
  });

  it('routes done with stop reason and error', () => {
    const h = handlers();
    dispatchGenerateEvent({ event: 'done', stop_reason: 'eos', error: 'boom' }, h);
    expect(h.onDone).toHaveBeenCalledWith('eos', 'boom');
  });

  it('ignores unknown frames', () => {
    const h = handlers();
    dispatchGenerateEvent({ event: 'telemetry_v9' }, h);
    for (const fn of Object.values(h)) expect(fn).not.toHaveBeenCalled();
  });
});

describe('consumeGenerateStream', () => {
  it('drains all frames in order and propagates stream errors', async () => {
    const h = handlers();
    async function* frames(): AsyncGenerator<SseFrame> {
      yield { event: 'step', step: { step: 0 } };
      yield { event: 'done', stop_reason: 'length' };
      throw new Error('connection lost');
    }
    await expect(consumeGenerateStream(frames(), h)).rejects.toThrow('connection lost');
    expect(h.onStep).toHaveBeenCalledTimes(1);
    expect(h.onDone).toHaveBeenCalledWith('length', null);
  });
});
