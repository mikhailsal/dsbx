/**
 * Chat-template fetch + cache for the composer.
 *
 * One request per (backend, model) pair per page lifetime -- templates are
 * static per model, and the backend itself caches discovery, so refetching
 * on every toggle flip would only add latency. Failures are cached too
 * (as a degraded ``ChatTemplateResponse``-shaped value with the error in
 * ``note``) so a dead provider doesn't get hammered on every keystroke.
 */

import { apiFetch } from '$lib/api';
import type { ChatTemplateResponse } from '$lib/types';
import type { TemplateInputs } from './types';

const cache = new Map<string, Promise<ChatTemplateResponse>>();

/**
 * ``freshness`` is an extra cache-key component for backends whose
 * loaded model can change WITHOUT the (backend, model) pair changing --
 * remote / local backends where the Status page swaps the model while
 * the request's ``model`` field stays empty. Callers pass the backend's
 * ``loaded_model`` so a swap invalidates the cache instead of serving
 * the previous model's template for the rest of the page lifetime.
 */
export function fetchChatTemplate(
  backend: string,
  model: string | null,
  freshness = ''
): Promise<ChatTemplateResponse> {
  const key = `${backend}\u0000${model ?? ''}\u0000${freshness}`;
  let entry = cache.get(key);
  if (!entry) {
    const qs = new URLSearchParams({ backend });
    if (model) qs.set('model', model);
    entry = apiFetch<ChatTemplateResponse>(`/api/v1/chat/template?${qs.toString()}`).catch(
      (e: unknown) => {
        cache.delete(key);
        throw e;
      }
    );
    cache.set(key, entry);
  }
  return entry;
}

/**
 * The inputs the engine renders with. A base model (or failed discovery)
 * falls back to the server-provided generic ChatML scaffold so the
 * composer still works -- the banner explains the substitution.
 */
export function templateInputsOf(info: ChatTemplateResponse): TemplateInputs {
  return {
    template: info.template ?? info.fallback_template,
    bosToken: info.bos_token,
    eosToken: info.eos_token
  };
}
