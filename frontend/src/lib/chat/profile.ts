/**
 * Derive a ``TemplateProfile`` from a chat template -- the marker map that
 * powers raw-mode parsing and the smart snippet buttons.
 *
 * Strategy (pragmatic, per the plan): render a handful of SENTINEL
 * conversations through the real template and recover each role's
 * prefix/suffix by diffing where the sentinel contents land. The split
 * between "suffix of the previous turn" and "prefix of the next" is
 * resolved with longest-common-prefix comparisons against tail renders --
 * see the inline derivation notes. Derived markers are then merged with
 * curated per-family overrides (reasoning tags, tool-call wrappers,
 * family-specific notes), and finally validated with a round-trip
 * self-test: render a sentinel doc -> parse it back -> compare. Only a
 * profile that passes the self-test gets ``complete: true``; anything
 * else degrades honestly to raw-only editing.
 */

import { Template } from '@huggingface/jinja';
import { parseRaw } from './parse';
import type {
  ChatMessage,
  RoleMarkers,
  TemplateFamily,
  TemplateInputs,
  TemplateProfile
} from './types';

// Alphanumeric-only so `| trim` / whitespace-control templates can't
// mangle them, and weird enough to never collide with marker text.
const U1 = 'dsbxSentinelUserOne';
const U2 = 'dsbxSentinelUserTwo';
const A1 = 'dsbxSentinelAsstOne';
const S0 = 'dsbxSentinelSysZero';
const T1 = 'dsbxSentinelToolOne';

interface FamilySpec {
  fingerprint: string;
  reasoning: { open: string; close: string } | null;
  toolCall: { open: string; close: string } | null;
  notes: string[];
}

/**
 * Curated in-content conventions by family. Role markers are always
 * DERIVED (they follow from the template itself); what templates cannot
 * tell us mechanically is how a family embeds reasoning and tool calls
 * inside assistant content -- that knowledge is curated here.
 */
const FAMILIES: Record<Exclude<TemplateFamily, 'unknown'>, FamilySpec> = {
  llama3: {
    fingerprint: '<|start_header_id|>',
    reasoning: null,
    toolCall: null,
    notes: [
      'Llama-3 family: tool calls render as JSON (or <|python_tag|> code) inside the assistant turn.'
    ]
  },
  harmony: {
    fingerprint: '<|channel|>',
    reasoning: { open: '<|channel|>analysis<|message|>', close: '<|end|>' },
    toolCall: null,
    notes: [
      'gpt-oss Harmony format: assistant turns are split into channels ' +
        '(analysis = reasoning, final = the answer, commentary = tool calls).'
    ]
  },
  gemma: {
    fingerprint: '<start_of_turn>',
    reasoning: null,
    toolCall: null,
    notes: ['Gemma renders the assistant role as "model" and rejects system messages.']
  },
  mistral: {
    fingerprint: '[INST]',
    reasoning: null,
    toolCall: { open: '[TOOL_CALLS] ', close: '</s>' },
    notes: ['Mistral merges the system prompt into the last user message.']
  },
  chatml: {
    fingerprint: '<|im_start|>',
    reasoning: { open: '<think>', close: '</think>' },
    toolCall: { open: '<tool_call>\n', close: '\n</tool_call>' },
    notes: []
  }
};

export function detectFamily(template: string): TemplateFamily {
  const entries = Object.entries(FAMILIES) as [keyof typeof FAMILIES, FamilySpec][];
  for (const [family, spec] of entries) {
    if (template.includes(spec.fingerprint)) return family;
  }
  return 'unknown';
}

function render(
  source: string,
  inputs: TemplateInputs,
  messages: ChatMessage[],
  addGenerationPrompt: boolean
): string {
  const template = new Template(source);
  return template.render({
    messages,
    add_generation_prompt: addGenerationPrompt,
    bos_token: inputs.bosToken ?? '',
    eos_token: inputs.eosToken ?? ''
  });
}

function longestCommonPrefix(a: string, b: string): string {
  let i = 0;
  while (i < a.length && i < b.length && a[i] === b[i]) i++;
  return a.slice(0, i);
}

function emptyProfile(family: TemplateFamily, notes: string[]): TemplateProfile {
  return {
    family,
    preamble: '',
    roles: {},
    generationPrompt: '',
    systemSupported: false,
    complete: false,
    reasoning: null,
    toolCall: null,
    notes
  };
}

/**
 * Recover user/assistant markers from sentinel renders.
 *
 * Derivation notes -- with ``C = render([u:U1, a:A1])`` and
 * ``D = render([u:U1, a:A1, u:U2])`` (no generation prompt):
 * - the gap between U1 and A1 in ``D`` is ``userSuffix + assistantPrefix``.
 *   Taken from ``D`` (assistant MID-conversation), not ``C``: several
 *   templates render the LAST assistant turn specially (Qwen3/3.5 inject
 *   an empty ``<think>`` scaffold), which would contaminate the prefix;
 * - the tail after U2 in ``D`` is ``userSuffix`` (+ any unconditional
 *   trailer some templates always append);
 * - so ``userSuffix = lcp(gap, tail)`` and the remainder of the gap is
 *   ``assistantPrefix``. Symmetrically for the assistant side (the
 *   assistant SUFFIX is safe to take from ``C`` -- the special-casing
 *   lives before the content, not after). Templates that unconditionally
 *   append the assistant header (some Llama-3 variants) yield
 *   shifted-but-consistent markers -- the round-trip self-test is the
 *   arbiter of whether they parse correctly.
 */
function deriveCoreMarkers(
  source: string,
  inputs: TemplateInputs
): {
  preamble: string;
  user: RoleMarkers;
  assistant: RoleMarkers;
  generationPrompt: string;
} | null {
  const C = render(source, inputs, [
    { role: 'user', content: U1 },
    { role: 'assistant', content: A1 }
  ], false);
  const D = render(source, inputs, [
    { role: 'user', content: U1 },
    { role: 'assistant', content: A1 },
    { role: 'user', content: U2 }
  ], false);
  const idxU1 = D.indexOf(U1);
  const idxA1 = D.indexOf(A1);
  const idxU2 = D.indexOf(U2);
  const idxA1C = C.indexOf(A1);
  if (idxU1 === -1 || idxA1 === -1 || idxA1 < idxU1 || idxU2 === -1 || idxA1C === -1) {
    return null;
  }

  const head = D.slice(0, idxU1);
  const gapUA = D.slice(idxU1 + U1.length, idxA1);
  const tailA = C.slice(idxA1C + A1.length);
  const gapAU = D.slice(idxA1 + A1.length, idxU2);
  const tailU = D.slice(idxU2 + U2.length);

  const userSuffix = longestCommonPrefix(gapUA, tailU);
  const assistantPrefix = gapUA.slice(userSuffix.length);
  const assistantSuffix = longestCommonPrefix(gapAU, tailA);
  const userPrefix = gapAU.slice(assistantSuffix.length);
  const preamble = head.endsWith(userPrefix)
    ? head.slice(0, head.length - userPrefix.length)
    : head;

  // Generation prompt = what add_generation_prompt appends on top of the
  // same conversation. Empty for templates that always append it.
  const E = render(source, inputs, [{ role: 'user', content: U1 }], false);
  const F = render(source, inputs, [{ role: 'user', content: U1 }], true);
  const generationPrompt = F.startsWith(E) ? F.slice(E.length) : '';

  return {
    preamble,
    user: { prefix: userPrefix, suffix: userSuffix },
    assistant: { prefix: assistantPrefix, suffix: assistantSuffix },
    generationPrompt
  };
}

/** Derive system markers; returns markers, "merged" (Mistral-style), or
 * "unsupported" (Gemma-style raise_exception). */
function deriveSystemMarkers(
  source: string,
  inputs: TemplateInputs,
  core: { preamble: string; user: RoleMarkers }
): { status: 'ok'; markers: RoleMarkers } | { status: 'merged' } | { status: 'unsupported' } {
  let E: string;
  try {
    E = render(source, inputs, [
      { role: 'system', content: S0 },
      { role: 'user', content: U1 },
      { role: 'assistant', content: A1 }
    ], false);
  } catch {
    return { status: 'unsupported' };
  }
  const idxS = E.indexOf(S0);
  const idxU = E.indexOf(U1);
  if (idxS === -1) return { status: 'unsupported' };
  if (idxU !== -1 && idxS > idxU) return { status: 'merged' };

  let prefix = E.slice(0, idxS);
  if (prefix.startsWith(core.preamble)) prefix = prefix.slice(core.preamble.length);
  const gapSU = E.slice(idxS + S0.length, idxU === -1 ? E.length : idxU);
  const suffix = gapSU.endsWith(core.user.prefix)
    ? gapSU.slice(0, gapSU.length - core.user.prefix.length)
    : gapSU;
  return { status: 'ok', markers: { prefix, suffix } };
}

/** Best-effort tool-result role markers (many templates reject role=tool).
 * The trailing user turn keeps the post-tool assistant turn MID-conversation
 * so last-turn scaffolds (Qwen `<think>`) don't leak into the suffix. */
function deriveToolMarkers(
  source: string,
  inputs: TemplateInputs,
  core: { assistant: RoleMarkers }
): RoleMarkers | null {
  let E: string;
  try {
    E = render(source, inputs, [
      { role: 'user', content: U1 },
      { role: 'assistant', content: A1 },
      { role: 'tool', content: T1 },
      { role: 'assistant', content: 'dsbxSentinelAsstTwo' },
      { role: 'user', content: U2 }
    ], false);
  } catch {
    return null;
  }
  const idxT = E.indexOf(T1);
  if (idxT === -1) return null;
  const idxA1end = E.indexOf(A1) + A1.length;
  const gapAT = E.slice(idxA1end, idxT);
  const prefix = gapAT.startsWith(core.assistant.suffix)
    ? gapAT.slice(core.assistant.suffix.length)
    : gapAT;
  const idxA2 = E.indexOf('dsbxSentinelAsstTwo');
  const gapTA = E.slice(idxT + T1.length, idxA2 === -1 ? E.length : idxA2);
  const suffix = gapTA.endsWith(core.assistant.prefix)
    ? gapTA.slice(0, gapTA.length - core.assistant.prefix.length)
    : gapTA;
  return { prefix, suffix };
}

/** Round-trip self-test: render a sentinel conversation with the real
 * template, parse it back with the candidate profile, compare shapes. */
function selfTest(source: string, inputs: TemplateInputs, profile: TemplateProfile): boolean {
  const conv: ChatMessage[] = [
    { role: 'user', content: U1 },
    { role: 'assistant', content: A1 },
    { role: 'user', content: U2 }
  ];
  let raw: string;
  try {
    raw = render(source, inputs, conv, true);
  } catch {
    return false;
  }
  const result = parseRaw(raw, { ...profile, complete: true });
  if (!result.ok) return false;
  const got = result.doc.blocks
    .filter((b) => 'content' in b && b.content.trim())
    .map((b) => ('content' in b ? `${b.kind}:${b.content.trim()}` : ''));
  const want = [`user:${U1}`, `assistant:${A1}`, `user:${U2}`];
  return want.every((w, i) => got[i] === w);
}

/**
 * Build the full profile for a template. Never throws: templates the
 * derivation cannot handle produce ``complete: false`` with an
 * explanatory note (the UI then offers raw-only editing).
 */
export function deriveProfile(inputs: TemplateInputs): TemplateProfile {
  if (!inputs.template) {
    return emptyProfile('unknown', ['no chat template available; raw editing only.']);
  }
  const source = inputs.template;
  const family = detectFamily(source);
  const spec = family === 'unknown' ? null : FAMILIES[family];

  let core: ReturnType<typeof deriveCoreMarkers>;
  try {
    core = deriveCoreMarkers(source, inputs);
  } catch (e) {
    return emptyProfile(family, [
      `marker derivation failed: ${e instanceof Error ? e.message : String(e)}`,
      ...(spec?.notes ?? [])
    ]);
  }
  if (!core) {
    return emptyProfile(family, [
      'could not locate sentinel contents in the rendered template; raw editing only.',
      ...(spec?.notes ?? [])
    ]);
  }

  const profile: TemplateProfile = {
    family,
    preamble: core.preamble,
    roles: { user: core.user, assistant: core.assistant },
    generationPrompt: core.generationPrompt,
    systemSupported: false,
    complete: false,
    reasoning: spec?.reasoning ?? null,
    toolCall: spec?.toolCall ?? null,
    notes: [...(spec?.notes ?? [])]
  };

  const sys = deriveSystemMarkers(source, inputs, core);
  if (sys.status === 'ok') {
    profile.roles.system = sys.markers;
    profile.systemSupported = true;
  } else if (sys.status === 'merged') {
    profile.systemSupported = true;
    profile.notes.push('system prompt is merged into a user turn by this template (no standalone marker).');
  } else {
    profile.notes.push('this template rejects system messages.');
  }

  const tool = deriveToolMarkers(source, inputs, core);
  if (tool && (tool.prefix || tool.suffix)) profile.roles.tool = tool;

  profile.complete = selfTest(source, inputs, profile);
  if (!profile.complete) {
    profile.notes.push(
      'round-trip self-test failed for this template; Blocks mode may not parse raw text reliably.'
    );
  }
  return profile;
}
