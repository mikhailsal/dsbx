<script lang="ts">
  /**
   * "What the model actually sees" panel: the Jinja chat-template source,
   * where it came from, and the special tokens the tokenizer names.
   * Collapsed by default -- it's reference material, not a control.
   */
  import type { ChatTemplateResponse } from '$lib/types';

  interface Props {
    info: ChatTemplateResponse;
  }
  let { info }: Props = $props();

  const sourceLabels: Record<string, string> = {
    hf_hub: 'HuggingFace Hub (tokenizer_config.json of the mapped repo)',
    gguf: 'GGUF metadata (tokenizer.chat_template)',
    transformers: 'local transformers tokenizer',
    remote: 'remote dsbx-serve host',
    none: 'no template found'
  };

  let specialEntries = $derived(Object.entries(info.special_tokens));
</script>

<details class="rounded border border-slate-800 bg-slate-900/40">
  <summary class="cursor-pointer px-3 py-2 text-xs text-slate-400 hover:text-slate-200 select-none">
    chat template
    <span class="ml-2 font-mono text-[10px] text-slate-500">
      {info.template ? `${info.template.length} chars` : 'fallback (ChatML)'}
      · {sourceLabels[info.source] ?? info.source}
    </span>
  </summary>
  <div class="px-3 pb-3 space-y-2">
    <p class="text-[11px] text-slate-500 leading-snug">
      This Jinja program ships with the model's tokenizer and turns a
      structured conversation into the single raw token stream the model
      actually consumes. The composer runs it in your browser — chat is
      just text, and this is the program that writes it.
    </p>
    {#if specialEntries.length}
      <div class="flex flex-wrap gap-x-4 gap-y-1 text-[11px]">
        {#each specialEntries as [name, text] (name)}
          <span class="text-slate-500">
            {name}: <span class="font-mono text-slate-300">{text}</span>
          </span>
        {/each}
      </div>
    {/if}
    <pre class="max-h-64 overflow-auto rounded border border-slate-800/60 bg-slate-950/60 p-2 font-mono text-[11px] leading-relaxed text-slate-300 whitespace-pre-wrap">{info.template ?? info.fallback_template}</pre>
  </div>
</details>
