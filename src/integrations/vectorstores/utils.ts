/**
 * Shared utilities for vectorstore integrations.
 * Provides emit_vector_trace and the sync/async wrapper factories,
 * and summarization helpers used by all vectorstore patches.
 */
import { logTrace } from "../../tracer";
import { TraceTypeValue } from "../shared/models";
import { currentParentId } from "../shared/context";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

export const MAX_IDS = 32;
export const MAX_STR = 256;
export const MAX_MATCHES = 10;
export const MAX_CHUNK_CHARS = 2000;
export const MAX_QUERY_CHARS = 4000;
export const MAX_METADATA_CHARS = 1000;

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

export function truncateText(value: unknown, limit: number): unknown {
  if (value == null || typeof value !== "string") return value;
  if (value.length <= limit) return value;
  return value.slice(0, limit) + "...[truncated]";
}

export function safeJsonable(obj: unknown, maxStr = MAX_STR): unknown {
  if (obj == null) return null;
  let s: string;
  try {
    s = JSON.stringify(obj, (_k, v) => (typeof v === "bigint" ? String(v) : v));
  } catch {
    try {
      s = JSON.stringify(String(obj));
    } catch {
      return { type: typeof obj };
    }
  }
  if (s.length > maxStr) s = s.slice(0, maxStr) + "...[truncated]";
  try {
    return JSON.parse(s);
  } catch {
    return s;
  }
}

export function captureQueryTexts(texts: unknown): unknown[] | null {
  if (texts == null) return null;
  if (typeof texts === "string") return [truncateText(texts, MAX_QUERY_CHARS)];
  if (Array.isArray(texts)) {
    return texts
      .filter((t) => t != null)
      .map((t) => truncateText(t, MAX_QUERY_CHARS));
  }
  return [truncateText(String(texts), MAX_QUERY_CHARS)];
}

export function captureMetadata(meta: unknown): unknown {
  if (meta == null) return null;
  return safeJsonable(meta, MAX_METADATA_CHARS);
}

export function truncateIds(ids: unknown, maxItems = MAX_IDS): { count: number; sample: string[]; truncated: boolean } | null {
  if (ids == null) return null;
  let seq: unknown[];
  try {
    seq = Array.from(ids as Iterable<unknown>);
  } catch {
    return null;
  }
  const total = seq.length;
  return {
    count: total,
    sample: seq.slice(0, maxItems).map(String),
    truncated: total > maxItems,
  };
}

export function vectorDim(vec: unknown): number | null {
  if (vec == null) return null;
  try {
    return (vec as unknown[]).length;
  } catch {
    return null;
  }
}

export function vectorCountAndDim(vectors: unknown): { count: number | null; dim: number | null } {
  if (vectors == null) return { count: null, dim: null };
  let seq: unknown[];
  try {
    seq = Array.from(vectors as Iterable<unknown>);
  } catch {
    return { count: null, dim: null };
  }
  const count = seq.length;
  let dim: number | null = null;
  if (seq.length > 0) {
    const first = seq[0];
    if (Array.isArray(first)) {
      dim = vectorDim(first);
    } else {
      dim = vectorDim(seq);
      return { count: 1, dim };
    }
  }
  return { count, dim };
}

export function buildMatch(opts: {
  id?: unknown;
  score?: unknown;
  text?: unknown;
  metadata?: unknown;
}): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  if (opts.id != null) out["id"] = String(opts.id);
  if (typeof opts.score === "number") out["score"] = opts.score;
  if (opts.text != null) out["text"] = truncateText(opts.text, MAX_CHUNK_CHARS);
  if (opts.metadata != null) out["metadata"] = captureMetadata(opts.metadata);
  return out;
}

export function captureMatches(
  matches: Record<string, unknown>[]
): { items: Record<string, unknown>[]; count: number; truncated: boolean } | null {
  if (!matches || matches.length === 0) return null;
  const total = matches.length;
  const head = matches.slice(0, MAX_MATCHES);
  return { items: head, count: total, truncated: total > MAX_MATCHES };
}

// ---------------------------------------------------------------------------
// emitVectorTrace
// ---------------------------------------------------------------------------

export function emitVectorTrace(opts: {
  integration: TraceTypeValue;
  api: string;
  target?: unknown;
  query?: unknown;
  result?: unknown;
  mutation?: unknown;
  start: number;
  end: number;
  success?: boolean;
  error?: string | null;
}): void {
  try {
    const payload: Record<string, unknown> = {
      type: "vectorstore",
      integration: opts.integration,
      api: opts.api,
      latency: opts.end - opts.start,
      parent_id: currentParentId(),
      success: opts.success ?? true,
    };
    if (opts.target != null) payload["target"] = opts.target;
    if (opts.query != null) payload["query"] = opts.query;
    if (opts.result != null) payload["result"] = opts.result;
    if (opts.mutation != null) payload["mutation"] = opts.mutation;
    if (opts.error != null) payload["error"] = opts.error;
    logTrace(payload).catch(() => {});
  } catch {
    // fail open
  }
}

type Summarizer = (args: unknown[], kwargs: Record<string, unknown>, instance: unknown, response?: unknown) => Record<string, unknown>;

function _safeSummarize(
  fn: Summarizer,
  args: unknown[],
  kwargs: Record<string, unknown>,
  instance: unknown,
  response?: unknown
): Record<string, unknown> {
  try {
    return fn(args, kwargs, instance, response) ?? {};
  } catch {
    return {};
  }
}

// ---------------------------------------------------------------------------
// Wrapper factories
// ---------------------------------------------------------------------------

/** Wraps a method with caching (query ops that can be served from Redis). */
export function makeSyncWrapper(
  original: (this: unknown, ...args: unknown[]) => unknown,
  integration: TraceTypeValue,
  api: string,
  summarize: Summarizer
): (this: unknown, ...args: unknown[]) => unknown {
  function wrapped(this: unknown, ...args: unknown[]): unknown {
    const kwargs = (args.length > 0 && args[args.length - 1] !== null && typeof args[args.length - 1] === "object" && !Array.isArray(args[args.length - 1])
      ? args[args.length - 1]
      : {}) as Record<string, unknown>;
    const positional = kwargs === args[args.length - 1] ? args.slice(0, -1) : args;

    const start = Date.now() / 1000;
    let response: unknown;
    try {
      response = original.call(this, ...args);
    } catch (e) {
      const end = Date.now() / 1000;
      const summary = _safeSummarize(summarize, positional, kwargs, this);
      emitVectorTrace({ integration, api, ...summary, start, end, success: false, error: (e as Error).constructor.name });
      throw e;
    }
    const end = Date.now() / 1000;
    const summary = _safeSummarize(summarize, positional, kwargs, this, response);
    emitVectorTrace({ integration, api, ...summary, start, end });
    return response;
  }
  return wrapped;
}

export function makeAsyncWrapper(
  original: (this: unknown, ...args: unknown[]) => Promise<unknown>,
  integration: TraceTypeValue,
  api: string,
  summarize: Summarizer
): (this: unknown, ...args: unknown[]) => Promise<unknown> {
  async function wrapped(this: unknown, ...args: unknown[]): Promise<unknown> {
    const kwargs = (args.length > 0 && args[args.length - 1] !== null && typeof args[args.length - 1] === "object" && !Array.isArray(args[args.length - 1])
      ? args[args.length - 1]
      : {}) as Record<string, unknown>;
    const positional = kwargs === args[args.length - 1] ? args.slice(0, -1) : args;

    const start = Date.now() / 1000;
    let response: unknown;
    try {
      response = await original.call(this, ...args);
    } catch (e) {
      const end = Date.now() / 1000;
      const summary = _safeSummarize(summarize, positional, kwargs, this);
      emitVectorTrace({ integration, api, ...summary, start, end, success: false, error: (e as Error).constructor.name });
      throw e;
    }
    const end = Date.now() / 1000;
    const summary = _safeSummarize(summarize, positional, kwargs, this, response);
    emitVectorTrace({ integration, api, ...summary, start, end });
    return response;
  }
  return wrapped;
}
