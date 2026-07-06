/**
 * "Append as assistant" as a pure ``ChatDoc`` transform -- the chat-mode
 * sibling of Text mode's "move to prompt", extracted from the composer so
 * the merge / close rules are unit-testable.
 *
 * The two cases are deliberately asymmetric:
 *
 * - an UNFINISHED (truncated) run leaves an OPEN prefill turn, kept
 *   VERBATIM -- open ``<think>`` tags included -- because the open turn
 *   is appended to the raw prompt as literal text and must reproduce the
 *   model's mid-turn context byte-for-byte;
 * - a FINISHED run CLOSES the turn, which templates re-render themselves
 *   (Qwen wraps ``reasoning_content`` in its own ``<think>`` scaffold),
 *   so the literal text is normalized first: the end-of-turn token the
 *   model emitted is stripped and reasoning sections become
 *   ``assistant_reasoning`` sub-blocks.
 */

import { splitAssistantContent } from './parse';
import { prefillMarkers } from './profile';
import type { ChatBlock, ChatDoc, TemplateProfile } from './types';

/** Drop the turn-closing text a finished run leaves at the end of the
 * completion (the EOS/EOT token the model emitted -- ``<|im_end|>``, or
 * Harmony's last-turn ``<|return|>``) so it is not doubled when the
 * template closes the turn again. */
function stripTurnClose(
  text: string,
  profile: TemplateProfile | null,
  eosToken: string | null
): string {
  const suffix = profile?.roles.assistant?.suffix ?? '';
  const candidates = [
    suffix,
    suffix.trimEnd(),
    profile?.lastAssistantSuffix ?? '',
    eosToken ?? ''
  ].filter(Boolean);
  for (const c of candidates) {
    if (text.endsWith(c)) return text.slice(0, -c.length);
  }
  return text;
}

/** The full turn's text -> blocks for a CLOSED turn (reasoning split out
 * so the template can re-render it per family convention). */
function closedTurnBlocks(
  turnText: string,
  profile: TemplateProfile | null,
  eosToken: string | null
): ChatBlock[] {
  const text = stripTurnClose(turnText, profile, eosToken);
  if (!profile) return [{ kind: 'assistant', content: text }];
  const blocks: ChatBlock[] = [];
  splitAssistantContent(blocks, [], profile, text);
  return blocks;
}

/**
 * Fold a run's streamed text into the conversation.
 *
 * - If the doc already ends in an OPEN prefill turn, the text is the
 *   model's continuation of that very turn: it merges into the same
 *   block, never splitting the assistant message in two.
 * - Otherwise a new turn starts. When the doc cued the model
 *   (``addGenerationPrompt``), the run began inside whatever scaffold the
 *   generation prompt carries beyond the turn opener (Qwen3.5's
 *   "<think>\n") -- that scaffold is folded into the turn text so the
 *   student sees it, tags included.
 * - ``finished`` (natural end-of-turn) closes the turn as described
 *   above; a truncated run leaves it open for the model to continue.
 */
export function appendAssistantText(
  doc: ChatDoc,
  text: string,
  finished: boolean,
  profile: TemplateProfile | null,
  eosToken: string | null = null
): ChatDoc {
  if (!text) return doc;
  const last = doc.blocks[doc.blocks.length - 1];
  const merging = last?.kind === 'assistant' && !!last.prefill;
  const scaffold = prefillMarkers(profile)?.scaffold ?? '';
  const turnText = merging
    ? last.content + text
    : (doc.addGenerationPrompt ? scaffold : '') + text;
  const prior = merging ? doc.blocks.slice(0, -1) : doc.blocks;
  const turnBlocks: ChatBlock[] = finished
    ? closedTurnBlocks(turnText, profile, eosToken)
    : [{ kind: 'assistant', content: turnText, prefill: true }];
  return { ...doc, blocks: [...prior, ...turnBlocks], addGenerationPrompt: false };
}
