/**
 * SSE frame dispatch for ``/api/v1/generate/stream``, extracted from
 * ``routes/generate/+page.svelte``. One tested place understands the
 * wire frames (``step`` / ``prompt_score`` / ``perf`` / ``raw_output``
 * / ``usage`` / ``done``); the page only wires handlers to state.
 */

import type { SseFrame } from '$lib/sse';
import type {
  GenStep,
  PerfMetricsPayload,
  RawOutputPayload,
  StepResult,
  UsagePayload
} from '$lib/types';

export interface GenerateStreamHandlers {
  onStep: (gs: GenStep) => void;
  onPromptScore: (steps: StepResult[], note: string) => void;
  onPerf: (metrics: PerfMetricsPayload | null) => void;
  onRawOutput: (payload: RawOutputPayload | null) => void;
  onUsage: (usage: UsagePayload) => void;
  onDone: (stopReason: string | null, error: string | null) => void;
}

/** Normalize a wire ``usage`` frame into the typed payload (missing
 * counters become explicit nulls so the UI renders em-dashes, not
 * ``undefined``). */
function usageOf(frame: SseFrame): UsagePayload {
  const u = frame as unknown as Partial<UsagePayload>;
  return {
    requests: u.requests ?? 0,
    prompt_tokens: u.prompt_tokens ?? null,
    completion_tokens: u.completion_tokens ?? null,
    total_tokens: u.total_tokens ?? null,
    notes: Array.isArray(u.notes) ? u.notes : []
  };
}

/** Route one SSE frame to its handler. Unknown events are ignored on
 * purpose -- a newer server adding frames must not break older UIs. */
export function dispatchGenerateEvent(evt: SseFrame, handlers: GenerateStreamHandlers): void {
  if (evt.event === 'step') {
    handlers.onStep(evt.step as GenStep);
  } else if (evt.event === 'prompt_score') {
    const ps = (evt as { steps?: StepResult[] }).steps ?? [];
    const note = (evt as { note?: string }).note ?? '';
    handlers.onPromptScore(ps, note);
  } else if (evt.event === 'perf') {
    const p = (evt as { metrics?: PerfMetricsPayload }).metrics;
    handlers.onPerf(p && typeof p === 'object' ? p : null);
  } else if (evt.event === 'raw_output') {
    const p = (evt as { payload?: RawOutputPayload }).payload;
    handlers.onRawOutput(p && typeof p === 'object' ? p : null);
  } else if (evt.event === 'usage') {
    handlers.onUsage(usageOf(evt));
  } else if (evt.event === 'done') {
    const stopReason = (evt as { stop_reason?: string | null }).stop_reason ?? null;
    const error = (evt as { error?: string | null }).error ?? null;
    handlers.onDone(stopReason, error);
  }
}

/** Drain a generate-stream's frames into the handlers. Exceptions from
 * the underlying stream propagate to the caller (which owns the busy
 * flag and the error toast). */
export async function consumeGenerateStream(
  events: AsyncIterable<SseFrame>,
  handlers: GenerateStreamHandlers
): Promise<void> {
  for await (const evt of events) {
    dispatchGenerateEvent(evt, handlers);
  }
}
