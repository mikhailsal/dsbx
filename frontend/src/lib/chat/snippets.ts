/**
 * Raw-mode smart buttons, generated from the ``TemplateProfile``.
 *
 * Because every snippet is built from markers DERIVED from the model's own
 * template (see ``profile.ts``), the buttons are model-correct by
 * construction: pick a Llama-3 model and "User turn" inserts
 * ``<|start_header_id|>user<|end_header_id|>``, pick Qwen and the same
 * button inserts ``<|im_start|>user`` -- no per-model if-chains anywhere.
 */

import type { Snippet, TemplateInputs, TemplateProfile } from './types';

function pairSnippet(
  id: string,
  label: string,
  open: string,
  close: string,
  title: string
): Snippet {
  return { id, label, text: open + close, cursorBack: close.length, title };
}

/**
 * Build the snippet list for the raw editor. Buttons appear only when the
 * template actually has the corresponding construct -- a family with no
 * reasoning convention gets no "thinking" button rather than a wrong one.
 */
export function buildSnippets(profile: TemplateProfile, inputs: TemplateInputs): Snippet[] {
  const snippets: Snippet[] = [];

  if (profile.preamble) {
    snippets.push({
      id: 'preamble',
      label: 'Preamble',
      text: profile.preamble,
      cursorBack: 0,
      title: 'What the template writes before the first message (usually BOS).'
    });
  }
  if (profile.systemSupported && profile.roles.system) {
    const { prefix, suffix } = profile.roles.system;
    snippets.push(
      pairSnippet('system', 'System', prefix, suffix, 'Open a system turn; cursor lands inside.')
    );
  }
  if (profile.roles.user) {
    const { prefix, suffix } = profile.roles.user;
    snippets.push(
      pairSnippet('user', 'User turn', prefix, suffix, 'Open a user turn; cursor lands inside.')
    );
  }
  if (profile.roles.assistant) {
    const { prefix, suffix } = profile.roles.assistant;
    snippets.push(
      pairSnippet(
        'assistant',
        'Assistant turn',
        prefix,
        suffix,
        'Author a complete assistant turn (respond on behalf of the AI).'
      )
    );
  }
  if (profile.generationPrompt) {
    snippets.push({
      id: 'generation-prompt',
      label: 'Cue model',
      text: profile.generationPrompt,
      cursorBack: 0,
      title:
        'The generation prompt: opens a fresh assistant turn for the MODEL ' +
        'to fill. Put it last (anything typed after it becomes a prefill).'
    });
  }
  if (profile.reasoning) {
    snippets.push(
      pairSnippet(
        'thinking',
        'Thinking',
        profile.reasoning.open,
        profile.reasoning.close,
        'Reasoning section inside an assistant turn (family convention).'
      )
    );
  }
  if (profile.toolCall) {
    const skeleton = '{"name": "", "arguments": {}}';
    snippets.push({
      id: 'tool-call',
      label: 'Tool call',
      text: profile.toolCall.open + skeleton + profile.toolCall.close,
      cursorBack: profile.toolCall.close.length + skeleton.length - '{"name": "'.length,
      title: 'Native tool-call skeleton inside an assistant turn (family convention).'
    });
  }
  if (profile.roles.tool) {
    const { prefix, suffix } = profile.roles.tool;
    snippets.push(
      pairSnippet('tool-result', 'Tool result', prefix, suffix, 'Feed a tool response back to the model.')
    );
  }
  if (inputs.eosToken) {
    snippets.push({
      id: 'eos',
      label: inputs.eosToken,
      text: inputs.eosToken,
      cursorBack: 0,
      title: 'The end-of-turn / end-of-sequence token.'
    });
  }
  return snippets;
}
