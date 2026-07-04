<script lang="ts">
  /**
   * Warning banner for base (non-chat-trained) models and for degraded
   * template discovery. Two distinct honest messages:
   *
   * - ``is_base_model``: discovery SUCCEEDED and found no template -- the
   *   model was never trained on chat formatting. We still let the user
   *   experiment (that's the pedagogical point) with the generic ChatML
   *   fallback, clearly labeled as our scaffold, not the model's.
   * - ``note`` set: discovery FAILED (network, gated repo, old server) --
   *   we don't know whether the model is chat-trained, and we say so.
   */
  import type { ChatTemplateResponse } from '$lib/types';

  interface Props {
    info: ChatTemplateResponse;
  }
  let { info }: Props = $props();
</script>

{#if info.is_base_model}
  <div class="rounded border border-amber-600/40 bg-amber-950/30 px-3 py-2 text-xs text-amber-300 leading-snug">
    <span class="font-semibold uppercase tracking-wider text-[10px]">base model</span>
    — this model ships <em>no chat template</em>, i.e. it was (almost
    certainly) never trained on chat formatting. The composer below uses a
    <span class="font-mono">generic ChatML</span> scaffold so you can
    experiment with what happens when you impose a chat structure on a
    plain text-continuation model — expect the model to imitate the
    format at best, and glitch at worst. That contrast is the lesson.
  </div>
{:else if info.note}
  <div class="rounded border border-slate-600/40 bg-slate-900/40 px-3 py-2 text-xs text-slate-400 leading-snug">
    <span class="font-semibold uppercase tracking-wider text-[10px]">template discovery failed</span>
    — {info.note}. Falling back to a generic ChatML scaffold; the real
    model may use different markers.
  </div>
{/if}
