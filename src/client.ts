import axios from "axios";
import { _config, authHeaders } from "./config";

// Track in-flight POSTs so short-lived scripts can await delivery before they
// exit. Most traces are emitted fire-and-forget (logTrace(...).catch()); a
// process that ends right after the last emit can otherwise terminate before
// the final requests reach the backend.
const _inflight = new Set<Promise<unknown>>();

function _track<T>(p: Promise<T>): Promise<T> {
  _inflight.add(p);
  p.then(
    () => _inflight.delete(p),
    () => _inflight.delete(p)
  );
  return p;
}

/**
 * Awaits all in-flight event POSTs (bounded by `timeoutMs`). Used internally by
 * the automatic `beforeExit` drain registered in instrument(), so short-lived
 * scripts deliver their final fire-and-forget traces without any manual call.
 */
export async function flushEvents(timeoutMs = 10000): Promise<void> {
  if (_inflight.size === 0) return;
  await Promise.race([
    Promise.allSettled([..._inflight]),
    new Promise((resolve) => setTimeout(resolve, timeoutMs)),
  ]);
}

export async function sendEvent(data: Record<string, unknown>): Promise<void> {
  if (!_config.enabled) return;

  try {
    const url = `${_config.endpoint}/${_config.version}/ingest`;
    await _track(
      axios.post(url, { event: data }, { headers: authHeaders(), timeout: 5000 })
    );
  } catch {
    // Fail open — never crash the caller's application.
  }
}

/**
 * Send event to /ingest and return the parsed response body.
 *
 * Used when `fluiq.secure({ mode: "block" })` is active so the caller can
 * inspect `response_blocked` before returning the LLM output to the user.
 * Falls back to `{}` on any network or parse error (fail-open).
 */
export async function sendEventGated(
  data: Record<string, unknown>
): Promise<Record<string, unknown>> {
  if (!_config.enabled) return {};
  try {
    const url = `${_config.endpoint}/${_config.version}/ingest`;
    const resp = await _track(
      axios.post(url, { event: data }, { headers: authHeaders(), timeout: 5000 })
    );
    return (resp.data as Record<string, unknown>) ?? {};
  } catch {
    // Fail open — never crash the caller's application.
    return {};
  }
}
