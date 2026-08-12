/**
 * Qdrant vectorstore integration (@qdrant/js-client-rest).
 * Patches QdrantClient.search/query (cached), .upsert/delete (invalidating),
 * .retrieve/scroll (plain). The JS client's methods are async, so they are
 * wrapped with the async wrappers (there is no separate AsyncQdrantClient).
 */
import { TraceType } from "../shared/models";
import {
  buildMatch,
  captureMatches,
  safeJsonable,
  truncateIds,
  vectorDim,
  makeAsyncWrapper,
} from "./utils";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function _coll(args: unknown[], kwargs: Record<string, unknown>): unknown {
  return kwargs["collection_name"] ?? (args.length > 0 ? args[0] : null);
}

function _target(args: unknown[], kwargs: Record<string, unknown>): Record<string, unknown> {
  return { collection: _coll(args, kwargs) };
}

function _payloadText(payload: unknown): string | null {
  if (typeof payload !== "object" || payload === null) return null;
  const p = payload as Record<string, unknown>;
  for (const key of ["text", "content", "page_content", "chunk", "document"]) {
    if (typeof p[key] === "string") return p[key] as string;
  }
  return null;
}

function _extractPoints(response: unknown): unknown[] {
  if (response == null) return [];
  const r = response as Record<string, unknown>;
  const pts = r["points"] ?? (Array.isArray(response) ? response : null);
  return Array.isArray(pts) ? pts : [];
}

function _buildMatchesFromPoints(points: unknown[]): { items: Record<string, unknown>[]; scores: number[] } {
  const items: Record<string, unknown>[] = [];
  const scores: number[] = [];
  for (const p of points) {
    const pRec = p as Record<string, unknown>;
    const pid = pRec["id"] ?? null;
    const sc = pRec["score"] ?? null;
    const payload = pRec["payload"] ?? null;
    if (typeof sc === "number") scores.push(sc);
    items.push(buildMatch({ id: pid, score: sc as number | undefined, text: _payloadText(payload) ?? undefined, metadata: payload }));
  }
  return { items, scores };
}

function _summarizeSearch(
  args: unknown[],
  kwargs: Record<string, unknown>,
  _instance: unknown,
  response?: unknown
): Record<string, unknown> {
  // JS SDK options use `vector`/`filter`/`with_vector`; Python used
  // `query_vector`/`query_filter`/`with_vectors`. Accept either.
  const vec = kwargs["vector"] ?? kwargs["query_vector"];
  const out: Record<string, unknown> = {
    target: _target(args, kwargs),
    query: {
      top_k: kwargs["limit"],
      vector_dim: typeof vec === "string" ? null : vectorDim(vec),
      vector_name: typeof vec === "string" ? vec : null,
      filter: safeJsonable(kwargs["filter"] ?? kwargs["query_filter"]),
      with_payload: kwargs["with_payload"],
      with_vectors: kwargs["with_vector"] ?? kwargs["with_vectors"],
      score_threshold: kwargs["score_threshold"],
    },
  };
  if (response != null) {
    const pts = Array.isArray(response) ? response : _extractPoints(response);
    const { items, scores } = _buildMatchesFromPoints(pts);
    out["result"] = {
      matches: captureMatches(items),
      score_min: scores.length > 0 ? Math.min(...scores) : null,
      score_max: scores.length > 0 ? Math.max(...scores) : null,
    };
  }
  return out;
}

function _summarizeQueryPoints(
  args: unknown[],
  kwargs: Record<string, unknown>,
  _instance: unknown,
  response?: unknown
): Record<string, unknown> {
  const q = kwargs["query"];
  const vectorDimVal = Array.isArray(q) ? vectorDim(q) : null;
  const out: Record<string, unknown> = {
    target: _target(args, kwargs),
    query: {
      top_k: kwargs["limit"],
      vector_dim: vectorDimVal,
      using: kwargs["using"],
      filter: safeJsonable(kwargs["query_filter"]),
      with_payload: kwargs["with_payload"],
      with_vectors: kwargs["with_vectors"],
    },
  };
  const pts = _extractPoints(response);
  if (pts.length > 0) {
    const { items, scores } = _buildMatchesFromPoints(pts);
    out["result"] = {
      matches: captureMatches(items),
      score_min: scores.length > 0 ? Math.min(...scores) : null,
      score_max: scores.length > 0 ? Math.max(...scores) : null,
    };
  }
  return out;
}

function _summarizeUpsert(
  args: unknown[],
  kwargs: Record<string, unknown>,
  _instance: unknown,
  _response?: unknown
): Record<string, unknown> {
  const points = kwargs["points"];
  let count: number | null = null;
  let dim: number | null = null;
  const ids: unknown[] = [];
  if (points != null) {
    try {
      const seq = Array.from(points as Iterable<unknown>);
      count = seq.length;
      for (const p of seq) {
        const pRec = p as Record<string, unknown>;
        const pid = pRec["id"];
        if (pid != null) ids.push(pid);
        const vec = pRec["vector"];
        if (dim == null && Array.isArray(vec)) dim = vectorDim(vec);
      }
    } catch { /* ignore */ }
  }
  return {
    target: _target(args, kwargs),
    mutation: {
      vector_count: count,
      vector_dim: dim,
      ids: ids.length > 0 ? truncateIds(ids) : null,
    },
  };
}

function _summarizeRetrieve(
  args: unknown[],
  kwargs: Record<string, unknown>,
  _instance: unknown,
  response?: unknown
): Record<string, unknown> {
  const out: Record<string, unknown> = {
    target: _target(args, kwargs),
    query: {
      ids: truncateIds(kwargs["ids"]),
      with_payload: kwargs["with_payload"],
      with_vectors: kwargs["with_vectors"],
    },
  };
  if (Array.isArray(response)) out["result"] = { count: response.length };
  return out;
}

function _summarizeDelete(
  args: unknown[],
  kwargs: Record<string, unknown>,
  _instance: unknown,
  _response?: unknown
): Record<string, unknown> {
  return {
    target: _target(args, kwargs),
    mutation: { points_selector: safeJsonable(kwargs["points_selector"]) },
  };
}

function _summarizeScroll(
  args: unknown[],
  kwargs: Record<string, unknown>,
  _instance: unknown,
  response?: unknown
): Record<string, unknown> {
  const out: Record<string, unknown> = {
    target: _target(args, kwargs),
    query: {
      limit: kwargs["limit"],
      filter: safeJsonable(kwargs["scroll_filter"]),
      offset: kwargs["offset"],
    },
  };
  if (Array.isArray(response) && response.length > 0) {
    const pts = Array.isArray(response[0]) ? response[0] : [];
    out["result"] = { count: pts.length };
  }
  return out;
}

// ---------------------------------------------------------------------------
// Patch
// ---------------------------------------------------------------------------

function _patchAsync(cls: { prototype: Record<string, unknown> }): void {
  const proto = cls.prototype;

  for (const [attr, api, summarize] of [
    ["search", "search", _summarizeSearch],
    // JS SDK names this `query`; Python's is `query_points`. Patch whichever exists.
    ["query", "query", _summarizeQueryPoints],
    ["query_points", "query_points", _summarizeQueryPoints],
    ["upsert", "upsert", _summarizeUpsert],
    ["delete", "delete", _summarizeDelete],
  ] as [string, string, typeof _summarizeSearch][]) {
    if (attr in proto) {
      proto[attr] = makeAsyncWrapper(
        proto[attr] as (this: unknown, ...args: unknown[]) => Promise<unknown>,
        TraceType.Qdrant, api, summarize
      );
    }
  }

  for (const [attr, api, summarize] of [
    ["retrieve", "retrieve", _summarizeRetrieve],
    ["scroll", "scroll", _summarizeScroll],
  ] as [string, string, typeof _summarizeRetrieve][]) {
    if (attr in proto) {
      proto[attr] = makeAsyncWrapper(
        proto[attr] as (this: unknown, ...args: unknown[]) => Promise<unknown>,
        TraceType.Qdrant, api, summarize
      );
    }
  }
}

export function patchQdrant(): void {
  // The JS QdrantClient's methods are all async (return Promises), unlike
  // Python's sync QdrantClient — so it must be patched with the async wrappers.
  // There is no separate AsyncQdrantClient export in the JS SDK.
  let QdrantClient: { prototype: Record<string, unknown> } | undefined;
  try {
    QdrantClient = (require("@qdrant/js-client-rest") as { QdrantClient: { prototype: Record<string, unknown> } }).QdrantClient;
  } catch {
    try {
      QdrantClient = (require("qdrant-js") as { QdrantClient: { prototype: Record<string, unknown> } }).QdrantClient;
    } catch { /* qdrant not installed */ }
  }
  if (QdrantClient && QdrantClient.prototype) _patchAsync(QdrantClient);
}
