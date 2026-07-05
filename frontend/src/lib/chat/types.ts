/**
 * Shapes for the client-side chat-template engine.
 *
 * The engine's job: turn a structured conversation (``ChatBlock[]``) into
 * the raw token stream the model actually consumes (via the model's own
 * Jinja chat template, rendered in the browser with ``@huggingface/jinja``)
 * -- and back. Everything here is deliberately plain-JSON-friendly so the
 * Decode workbench can stash a ``ChatDoc`` in component state or
 * localStorage without ceremony.
 */

/** One editable card in the block editor. */
export type ChatBlock =
  | { kind: 'system'; content: string }
  | { kind: 'user'; content: string }
  /**
   * ``prefill`` marks an OPEN assistant turn: the turn's closing markers
   * are not rendered, so the model continues mid-turn from exactly this
   * text (the raw prompt becomes ``...generation prompt + content``).
   * Only meaningful on the FINAL block; "append as assistant" sets it
   * automatically when the source run stopped on max_tokens instead of a
   * natural end-of-turn.
   */
  | { kind: 'assistant'; content: string; prefill?: boolean }
  /**
   * Reasoning/thinking section of the FOLLOWING assistant block (or a
   * standalone thinking turn if no assistant block follows). Rendered
   * into the assistant message using the family's convention
   * (``<think>...</think>`` for Qwen/ChatML-R1, analysis channel for
   * Harmony); families with no known convention get a warning.
   */
  | { kind: 'assistant_reasoning'; content: string }
  /**
   * The tool catalogue advertised to the model -- a JSON array of
   * OpenAI-style function definitions. Fed to the template as ``tools``
   * and forwarded verbatim on the simulation path.
   */
  | { kind: 'tool_defs'; toolsJson: string }
  /** A native tool invocation authored on behalf of the assistant. */
  | { kind: 'tool_call'; name: string; argumentsJson: string; callId?: string }
  /** The tool's reply, fed back to the model as a ``role: tool`` message. */
  | { kind: 'tool_result'; content: string; name?: string; callId?: string };

export type ChatBlockKind = ChatBlock['kind'];

/** The whole conversation as edited in the block UI. */
export interface ChatDoc {
  blocks: ChatBlock[];
  /**
   * Append the assistant header after the last message so the model
   * starts a fresh assistant turn (the standard "now you answer" cue).
   * Off when the last block is an assistant prefill the model should
   * continue mid-turn.
   */
  addGenerationPrompt: boolean;
}

/** OpenAI-shaped message -- what templates consume and what the
 * simulation path (`messages[]` on the generate wire) sends. */
export interface ChatMessage {
  role: string;
  content: string | null;
  name?: string;
  tool_calls?: ToolCallPayload[];
  tool_call_id?: string;
  /** Family-specific extras (e.g. ``reasoning_content``). */
  [key: string]: unknown;
}

export interface ToolCallPayload {
  id?: string;
  type: 'function';
  function: { name: string; arguments: string };
}

/** Template families the engine knows extra tricks for. */
export type TemplateFamily =
  | 'llama3'
  | 'chatml'
  | 'harmony'
  | 'gemma'
  | 'mistral'
  | 'unknown';

/** The literal text that opens / closes one role's turn. */
export interface RoleMarkers {
  prefix: string;
  suffix: string;
}

/**
 * Everything the raw-mode UX needs to know about a template, derived by
 * rendering sentinel conversations (see ``profile.ts``) and merged with
 * curated per-family overrides. ``complete`` gates the raw -> blocks
 * parser: when derivation could not recover reliable markers the parser
 * degrades to "one raw block + warning" instead of guessing.
 */
export interface TemplateProfile {
  family: TemplateFamily;
  /** Rendered before the first message (typically the BOS text). */
  preamble: string;
  roles: Partial<Record<'system' | 'user' | 'assistant' | 'tool', RoleMarkers>>;
  /** Appended when ``add_generation_prompt`` is true. */
  generationPrompt: string;
  systemSupported: boolean;
  complete: boolean;
  /** In-content reasoning tags (``<think>`` / ``</think>``), if any. */
  reasoning: { open: string; close: string } | null;
  /** In-content tool-call JSON wrapper (``<tool_call>`` style), if any. */
  toolCall: { open: string; close: string } | null;
  notes: string[];
}

/** What the template metadata endpoint gives the engine to work with. */
export interface TemplateInputs {
  template: string | null;
  bosToken: string | null;
  eosToken: string | null;
}

export interface RenderResult {
  /** The raw string to send as ``prompt`` (template-capable backends). */
  raw: string;
  /** The structured conversation for the simulation path. */
  messages: ChatMessage[];
  /** Parsed tool catalogue (from ``tool_defs``), or null. */
  tools: Record<string, unknown>[] | null;
  /** True when no model template existed and the ChatML fallback ran. */
  usedFallback: boolean;
  warnings: string[];
  /** Template execution failure (e.g. Gemma's "System role not
   * supported" raise_exception), verbatim from the Jinja engine.
   * ``raw`` is empty when set. */
  error: string | null;
}

/** A positioned, human-explainable parse failure -- never throws away
 * the user's text; the UI highlights ``position`` and shows ``hint``. */
export interface ParseError {
  position: number;
  expected: string;
  found: string;
  hint: string;
}

export interface ParseResult {
  ok: boolean;
  doc: ChatDoc;
  warnings: string[];
  error: ParseError | null;
}

/** One raw-mode smart button: insert ``text`` at the cursor, then move
 * the caret back by ``cursorBack`` characters (into the empty middle of
 * an open/close pair). */
export interface Snippet {
  id: string;
  label: string;
  text: string;
  cursorBack: number;
  title: string;
}
