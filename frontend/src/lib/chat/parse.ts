/**
 * Raw string -> blocks: the reverse half of the engine.
 *
 * Uses the ``TemplateProfile``'s derived role markers to segment a raw
 * prompt back into structured blocks so the user can switch Raw -> Blocks
 * without losing work. Deliberately pragmatic (per the plan): roles are
 * recovered generically from markers; reasoning sections and tool calls are
 * extracted for families with known in-content conventions; anything
 * unrecognized degrades to raw assistant text plus a warning -- and outright
 * structural mismatches return a positioned ``ParseError`` that never throws
 * away the user's text.
 */

import type {
  ChatBlock,
  ParseError,
  ParseResult,
  TemplateProfile
} from './types';

interface RoleEntry {
  role: 'system' | 'user' | 'assistant' | 'tool';
  prefix: string;
  suffix: string;
}

const KIND_BY_ROLE = {
  system: 'system',
  user: 'user',
  assistant: 'assistant',
  tool: 'tool_result'
} as const;

function roleEntries(profile: TemplateProfile): RoleEntry[] {
  const entries: RoleEntry[] = [];
  for (const role of ['system', 'user', 'assistant', 'tool'] as const) {
    const markers = profile.roles[role];
    if (markers) entries.push({ role, prefix: markers.prefix, suffix: markers.suffix });
  }
  // Longest prefix first: "<|im_start|>user\n<tool_response>\n" must win
  // over "<|im_start|>user\n", and a role with an EMPTY prefix (some
  // Llama-3 variants leave the assistant turn implicit after the user
  // suffix) sorts last as the catch-all.
  return entries.sort((a, b) => b.prefix.length - a.prefix.length);
}

/** Where the verbatim content of an OPEN assistant run starts: right
 * after the opener the renderer will prepend when re-rendering it (the
 * assistant prefix when it heads the generation prompt, else the
 * generation prompt itself -- mirrors ``prefillMarkers`` + ``renderChat``
 * so the raw <-> blocks loop stays byte-lossless). */
function openTurnContentStart(
  raw: string,
  turnPos: number,
  contentStart: number,
  profile: TemplateProfile
): number {
  const prefix = profile.roles.assistant?.prefix ?? '';
  const opener =
    prefix && profile.generationPrompt.startsWith(prefix) ? prefix : profile.generationPrompt;
  return opener && raw.startsWith(opener, turnPos) ? turnPos + opener.length : contentStart;
}

/** Where a turn with an EMPTY closing marker ends: at the next role
 * prefix (DeepSeek's ``<｜User｜>text<｜Assistant｜>`` -- nothing closes
 * the user turn, the next turn's opener delimits it), or -1 for "runs
 * to the end of the string". */
function nextRolePrefixAt(raw: string, from: number, entries: RoleEntry[]): number {
  let at = -1;
  for (const e of entries) {
    if (!e.prefix) continue;
    const i = raw.indexOf(e.prefix, from);
    if (i !== -1 && (at === -1 || i < at)) at = i;
  }
  return at;
}

function parseFailure(doc: { blocks: ChatBlock[] }, error: ParseError): ParseResult {
  return {
    ok: false,
    doc: { blocks: doc.blocks, addGenerationPrompt: false },
    warnings: [],
    error
  };
}

function preview(text: string, at: number): string {
  const snippet = text.slice(at, at + 40);
  return snippet.length ? JSON.stringify(snippet) : '<end of text>';
}

/**
 * Parse ``raw`` into a ``ChatDoc`` using the profile's markers.
 *
 * The error contract: ``ok=false`` carries a single positioned
 * ``ParseError`` (first structural mismatch) AND whatever blocks were
 * recovered before it, so the UI can highlight the offending position
 * while still showing the good part. ``warnings`` covers recoverable
 * oddities (unterminated final turn, unparseable tool-call JSON).
 */
export function parseRaw(raw: string, profile: TemplateProfile): ParseResult {
  const blocks: ChatBlock[] = [];
  const warnings: string[] = [];
  if (!profile.complete) {
    return parseFailure(
      { blocks: raw ? [{ kind: 'assistant', content: raw }] : [] },
      {
        position: 0,
        expected: 'a template with recoverable role markers',
        found: 'a template this engine could not derive markers from',
        hint:
          'The chat template is too unusual for round-trip parsing; ' +
          'edit in Raw mode (the text is preserved as a single block).'
      }
    );
  }

  // The preamble (BOS text, or a default system block some templates
  // inject when the conversation has none) is optional on parse: a raw
  // string that opens with an explicit role turn skips it legitimately.
  let pos = 0;
  if (profile.preamble && raw.startsWith(profile.preamble)) {
    pos = profile.preamble.length;
  }

  const entries = roleEntries(profile);
  // Tracks a run of CONSECUTIVE assistant-role turns (Harmony renders one
  // logical turn as several <|start|>assistant messages: analysis, then
  // final). If such a run ends UNTERMINATED, the whole run is folded back
  // into one verbatim open block -- split sub-blocks re-render through
  // the template, which reworks or drops non-final reasoning, so only the
  // literal text reproduces the mid-turn context. ``runOpenStart`` is
  // where the open block's content begins: right after the opener the
  // RENDERER will prepend when it re-renders the open turn (the assistant
  // prefix, the generation prompt, or -- for always-appended-header
  // templates -- the matched turn prefix), keeping the loop lossless.
  let runOpenStart = -1;
  let runBlockCount = -1;
  while (pos < raw.length) {
    // A bare generation prompt at the very tail means "model, your turn".
    // Checked as an EXACT tail match so it never shadows a real assistant
    // turn (in most templates the generation prompt IS the assistant
    // prefix); "generation prompt + prefill text" falls through to the
    // assistant role match below and parses as an open assistant turn.
    if (profile.generationPrompt && raw.slice(pos) === profile.generationPrompt) {
      return { ok: true, doc: { blocks, addGenerationPrompt: true }, warnings, error: null };
    }

    let entry = entries.find((e) => e.prefix && raw.startsWith(e.prefix, pos));
    if (!entry && profile.generationPrompt && raw.startsWith(profile.generationPrompt, pos)) {
      // Generation prompt + text = an assistant turn opened by the cue
      // (covers templates whose DERIVED assistant prefix carries extra
      // scaffolding, e.g. Qwen3's forced empty <think> block).
      entry = {
        role: 'assistant',
        prefix: profile.generationPrompt,
        suffix: profile.roles.assistant?.suffix ?? ''
      };
    }
    if (!entry) entry = entries.find((e) => e.prefix === '');
    if (!entry) {
      return parseFailure(
        { blocks },
        {
          position: pos,
          expected:
            'one of the role markers: ' +
            entries
              .filter((e) => e.prefix)
              .map((e) => `${e.role} ${JSON.stringify(e.prefix)}`)
              .join(', '),
          found: preview(raw, pos),
          hint:
            'Each turn must open with its role marker exactly as the ' +
            'template writes it -- check for typos in the special tokens.'
        }
      );
    }

    const contentStart = pos + entry.prefix.length;
    if (entry.role === 'assistant') {
      if (runOpenStart === -1) {
        runOpenStart = openTurnContentStart(raw, pos, contentStart, profile);
        runBlockCount = blocks.length;
      }
    } else {
      runOpenStart = -1;
    }

    // A role with an empty suffix (DeepSeek's user turns) is delimited
    // by the NEXT turn's opener instead of a closing marker of its own.
    const suffixAt = entry.suffix
      ? raw.indexOf(entry.suffix, contentStart)
      : nextRolePrefixAt(raw, contentStart, entries);
    if (suffixAt === -1) {
      const content = raw.slice(contentStart);
      if (!entry.suffix && entry.role !== 'assistant') {
        // Nothing left to delimit the turn and the role needs no closing
        // marker: a LEGITIMATELY closed final turn, not an open one.
        blocks.push({ kind: KIND_BY_ROLE[entry.role], content });
        return { ok: true, doc: { blocks, addGenerationPrompt: false }, warnings, error: null };
      }
      const lastClose = profile.lastAssistantSuffix;
      if (entry.role === 'assistant' && lastClose && content.endsWith(lastClose)) {
        // The FINAL assistant turn, closed with the template's last-turn
        // marker (Harmony's <|return|> where mid-stream turns use
        // <|end|>) -- a CLOSED turn, not a prefill.
        appendRoleContent(
          blocks, warnings, profile, 'assistant', content.slice(0, -lastClose.length)
        );
        return { ok: true, doc: { blocks, addGenerationPrompt: false }, warnings, error: null };
      }
      // Unterminated final turn -- normal when the user is mid-edit or
      // prefilling. The whole trailing assistant RUN (Harmony: closed
      // analysis turn + open final turn) folds into ONE verbatim open
      // block, mid-run markers included.
      if (entry.role === 'assistant') {
        blocks.length = runBlockCount;
        blocks.push({ kind: 'assistant', content: raw.slice(runOpenStart), prefill: true });
      } else {
        blocks.push({ kind: KIND_BY_ROLE[entry.role], content });
      }
      warnings.push(
        `final ${entry.role} turn is not closed with ${JSON.stringify(entry.suffix)}; kept as an open turn.`
      );
      return { ok: true, doc: { blocks, addGenerationPrompt: false }, warnings, error: null };
    }
    appendRoleContent(blocks, warnings, profile, entry.role, raw.slice(contentStart, suffixAt));
    pos = suffixAt + entry.suffix.length;
  }

  return { ok: true, doc: { blocks, addGenerationPrompt: false }, warnings, error: null };
}

function appendRoleContent(
  blocks: ChatBlock[],
  warnings: string[],
  profile: TemplateProfile,
  role: RoleEntry['role'],
  content: string
): void {
  if (role === 'assistant') {
    splitAssistantContent(blocks, warnings, profile, content);
    return;
  }
  blocks.push({ kind: KIND_BY_ROLE[role], content });
}

/**
 * Split one CLOSED assistant turn's content into reasoning / tool-call /
 * text blocks using the family's in-content conventions (``<think>``
 * tags, ``<tool_call>`` JSON wrappers). Unknown constructs stay as plain
 * assistant text -- with a warning when they LOOK structural.
 *
 * Exported because "append as assistant" reuses it when a FINISHED run
 * closes the turn: templates re-render closed turns themselves (Qwen
 * wraps ``reasoning_content`` in its own scaffold), so literal tags must
 * become sub-blocks to round-trip. Open (prefill) turns are the opposite
 * case and stay verbatim.
 */
export function splitAssistantContent(
  blocks: ChatBlock[],
  warnings: string[],
  profile: TemplateProfile,
  content: string
): void {
  let rest = content;

  if (profile.reasoning) {
    const { open, close } = profile.reasoning;
    const openAt = rest.indexOf(open);
    if (openAt !== -1) {
      const closeAt = rest.indexOf(close, openAt + open.length);
      const before = rest.slice(0, openAt);
      if (before.trim()) blocks.push({ kind: 'assistant', content: before });
      if (closeAt === -1) {
        blocks.push({ kind: 'assistant_reasoning', content: rest.slice(openAt + open.length) });
        // No warning when the reasoning close doubles as the turn suffix
        // (Harmony's <|end|>): the analysis turn is then LEGITIMATELY
        // closed by the turn marker the caller already consumed.
        if (close !== profile.roles.assistant?.suffix) {
          warnings.push(`reasoning section is missing its closing ${JSON.stringify(close)}.`);
        }
        return;
      }
      blocks.push({
        kind: 'assistant_reasoning',
        content: rest.slice(openAt + open.length, closeAt)
      });
      rest = rest.slice(closeAt + close.length).replace(/^\n+/, '');
      // Harmony closes the analysis MESSAGE and re-opens the assistant
      // header for the final channel; inside one logical turn that
      // header is scaffolding, not content -- drop it so the block holds
      // just the answer text.
      const reopen = profile.roles.assistant?.prefix ?? '';
      if (reopen && rest.startsWith(reopen)) rest = rest.slice(reopen.length);
    }
  }

  if (profile.toolCall) {
    const { open, close } = profile.toolCall;
    while (true) {
      const openAt = rest.indexOf(open);
      if (openAt === -1) break;
      const closeAt = rest.indexOf(close, openAt + open.length);
      if (closeAt === -1) {
        warnings.push(`tool call is missing its closing ${JSON.stringify(close)}; kept as text.`);
        break;
      }
      const before = rest.slice(0, openAt);
      if (before.trim()) blocks.push({ kind: 'assistant', content: before });
      pushToolCall(blocks, warnings, rest.slice(openAt + open.length, closeAt));
      rest = rest.slice(closeAt + close.length).replace(/^\n+/, '');
    }
  }

  if (rest) blocks.push({ kind: 'assistant', content: rest });
}

/** Parse a ``{"name": ..., "arguments": ...}`` tool-call body; malformed
 * JSON degrades to an assistant text block with a warning. */
function pushToolCall(blocks: ChatBlock[], warnings: string[], body: string): void {
  try {
    const parsed = JSON.parse(body) as { name?: unknown; arguments?: unknown };
    if (typeof parsed.name !== 'string') throw new Error('missing "name"');
    const args = parsed.arguments;
    blocks.push({
      kind: 'tool_call',
      name: parsed.name,
      argumentsJson: typeof args === 'string' ? args : JSON.stringify(args ?? {})
    });
  } catch (e) {
    warnings.push(
      `tool call body is not valid JSON (${e instanceof Error ? e.message : String(e)}); kept as text.`
    );
    blocks.push({ kind: 'assistant', content: body });
  }
}
