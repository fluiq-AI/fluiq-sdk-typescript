/**
 * Pinecone vectorstore integration (@pinecone-database/pinecone).
 * Patches Index.query (cached), Index.upsert/update/deleteOne/deleteMany/deleteAll
 * (invalidating), Index.fetch (plain). The JS SDK's Index methods are all async,
 * so they are wrapped with the async wrappers (there is no sync/async split as in
 * Python).
 */
import { TraceType } from "../shared/models";
import {
  buildMatch,
  captureMatches,
  safeJsonable,
  truncateIds,
  vectorDim,
  makeAsyncCachedWrapper,
  makeAsyncInvalidatingWrapper,
  makeAsyncWrapper,
} from "./utils";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function _pineconeTarget(
  _args: unknown[],
  _kwargs: Record<string, unknown>,
  instance: unknown
): string {
  const inst = instance as Record<string, unknown>;
  return String(
    inst["name"] ?? inst["_index_name"] ?? (inst["config"] as Record<string, unknown> | null)?.["name"] ?? ""
  );
}

function _target(instance: unknown, kwargs: Record<string, unknown>): Record<string, unknown> {
  const inst = instance as Record<string, unknown>;
  const name =
    inst["name"] ?? inst["_index_name"] ?? (inst["config"] as Record<string, unknown> | null)?.["name"] ?? null;
  return { index: name, namespace: kwargs["namespace"] };
}

function _vectorsSummary(vectors: unknown): { count: number | null; dim: number | null } {
  if (vectors == null) return { count: null, dim: null };
  let seq: unknown[];
  try { seq = Array.from(vectors as Iterable<unknown>); } catch { return { count: null, dim: null }; }
  const count = seq.length;
  let dim: number | null = null;
  if (seq.length > 0) {
    const first = seq[0] as Record<string, unknown>;
    const values = first["values"] ?? first["sparse_values"];
    dim = vectorDim(values);
  }
  return { count, dim };
}

function _summarizeQuery(
  _args: unknown[],
  kwargs: Record<string, unknown>,
  instance: unknown,
  response?: unknown
): Record<string, unknown> {
  const vec = kwargs["vector"];
  const out: Record<string, unknown> = {
    target: _target(instance, kwargs),
    query: {
      // JS uses topK / includeValues / includeMetadata; Python used snake_case.
      top_k: kwargs["topK"] ?? kwargs["top_k"],
      vector_dim: vectorDim(vec),
      has_vector: vec != null,
      id: kwargs["id"],
      filter: safeJsonable(kwargs["filter"]),
      include_values: kwargs["includeValues"] ?? kwargs["include_values"],
      include_metadata: kwargs["includeMetadata"] ?? kwargs["include_metadata"],
    },
  };

  if (response != null) {
    const rawMatches = (typeof response === "object"
      ? (response as Record<string, unknown>)["matches"] ?? null
      : null) as unknown[] | null;
    if (rawMatches != null) {
      const scores: number[] = [];
      const items = rawMatches.map((m) => {
        const isDict = typeof m === "object";
        const mRec = m as Record<string, unknown>;
        const mid = isDict ? mRec["id"] : null;
        const sc = isDict ? mRec["score"] : null;
        const md = isDict ? mRec["metadata"] : null;
        if (typeof sc === "number") scores.push(sc);
        let text: string | null = null;
        if (typeof md === "object" && md !== null) {
          const mdRec = md as Record<string, unknown>;
          for (const key of ["text", "content", "page_content", "chunk"]) {
            if (typeof mdRec[key] === "string") { text = mdRec[key] as string; break; }
          }
        }
        return buildMatch({ id: mid, score: sc as number | undefined, text: text ?? undefined, metadata: md });
      });
      out["result"] = {
        matches: captureMatches(items),
        score_min: scores.length > 0 ? Math.min(...scores) : null,
        score_max: scores.length > 0 ? Math.max(...scores) : null,
      };
    }
  }
  return out;
}

function _summarizeUpsert(
  args: unknown[],
  kwargs: Record<string, unknown>,
  instance: unknown,
  response?: unknown
): Record<string, unknown> {
  const vectors = kwargs["vectors"] ?? (args.length > 0 ? args[0] : null);
  const summary = _vectorsSummary(vectors);
  const out: Record<string, unknown> = {
    target: _target(instance, kwargs),
    mutation: { vector_count: summary.count, vector_dim: summary.dim },
  };
  if (response != null) {
    const r = response as Record<string, unknown>;
    const upserted = r["upserted_count"] ?? null;
    if (upserted != null) (out["mutation"] as Record<string, unknown>)["upserted_count"] = upserted;
  }
  return out;
}

function _summarizeFetch(
  args: unknown[],
  kwargs: Record<string, unknown>,
  instance: unknown,
  response?: unknown
): Record<string, unknown> {
  const ids = kwargs["ids"] ?? (args.length > 0 ? args[0] : null);
  const out: Record<string, unknown> = {
    target: _target(instance, kwargs),
    query: { ids: truncateIds(ids) },
  };
  if (response != null) {
    const r = response as Record<string, unknown>;
    const vectors = r["vectors"] ?? null;
    if (vectors != null) {
      try { out["result"] = { count: Object.keys(vectors as object).length }; } catch { /* ignore */ }
    }
  }
  return out;
}

function _summarizeDelete(
  args: unknown[],
  kwargs: Record<string, unknown>,
  instance: unknown,
  _response?: unknown
): Record<string, unknown> {
  // JS: deleteOne(id) | deleteMany(ids[] | filter). Python: delete({ ids, filter }).
  const mutation: Record<string, unknown> = {};
  const a0 = args.length > 0 ? args[0] : null;
  if (typeof a0 === "string" || typeof a0 === "number") {
    mutation["ids"] = truncateIds([a0]);
  } else if (Array.isArray(a0)) {
    mutation["ids"] = truncateIds(a0);
  } else if (a0 && typeof a0 === "object") {
    const o = a0 as Record<string, unknown>;
    if (o["ids"] != null || o["filter"] != null) {
      if (o["ids"] != null) mutation["ids"] = truncateIds(o["ids"]);
      if (o["filter"] != null) mutation["filter"] = safeJsonable(o["filter"]);
    } else {
      mutation["filter"] = safeJsonable(o);
    }
  }
  if (kwargs["ids"] != null) mutation["ids"] = truncateIds(kwargs["ids"]);
  if (kwargs["filter"] != null) mutation["filter"] = safeJsonable(kwargs["filter"]);
  return { target: _target(instance, kwargs), mutation };
}

function _summarizeDeleteAll(
  _args: unknown[],
  kwargs: Record<string, unknown>,
  instance: unknown,
  _response?: unknown
): Record<string, unknown> {
  return { target: _target(instance, kwargs), mutation: { delete_all: true } };
}

function _summarizeUpdate(
  _args: unknown[],
  kwargs: Record<string, unknown>,
  instance: unknown,
  _response?: unknown
): Record<string, unknown> {
  const vec = kwargs["values"];
  return {
    target: _target(instance, kwargs),
    mutation: {
      ids: truncateIds(kwargs["id"] ? [kwargs["id"]] : null),
      vector_dim: vectorDim(vec),
      has_metadata: kwargs["set_metadata"] != null,
    },
  };
}

async function _queryCacheKey(
  _args: unknown[],
  kwargs: Record<string, unknown>,
  instance: unknown
): Promise<string> {
  const inst = instance as Record<string, unknown>;
  const idx = String(
    inst["name"] ?? inst["_index_name"] ?? (inst["config"] as Record<string, unknown> | null)?.["name"] ?? ""
  );
  const { vectorstoreCacheKey } = require("../../optimization/client") as typeof import("../../optimization/client");
  return vectorstoreCacheKey(
    "pinecone",
    idx,
    kwargs["vector"] ?? kwargs["id"],
    (kwargs["topK"] ?? kwargs["top_k"]) as number,
    kwargs["filter"] as unknown
  );
}

function _queryMock(
  cachedResult: Record<string, unknown>,
  _args: unknown[],
  kwargs: Record<string, unknown>,
  _instance: unknown
): unknown {
  const items = ((cachedResult["matches"] as Record<string, unknown> | null)?.["items"] as Record<string, unknown>[] | null) ?? [];
  const matches = items.map((m) => ({
    id: m["id"] ?? "",
    score: m["score"] ?? null,
    metadata: m["metadata"] ?? null,
    values: [],
    sparse_values: null,
  }));
  return {
    matches,
    namespace: kwargs["namespace"] ?? "",
    usage: null,
  };
}

// ---------------------------------------------------------------------------
// Patch
// ---------------------------------------------------------------------------

// The JS SDK's Index methods are all async (return Promises). The single
// `Index` class covers every data operation; there is no sync/async split as
// in Python, and delete is split into deleteOne/deleteMany/deleteAll.
function _patchIndex(cls: { prototype: Record<string, unknown> }): void {
  const proto = cls.prototype;

  if (typeof proto["query"] === "function") {
    proto["query"] = makeAsyncCachedWrapper(
      proto["query"] as (this: unknown, ...args: unknown[]) => Promise<unknown>,
      TraceType.Pinecone, "query", _summarizeQuery, _queryCacheKey, _queryMock
    );
  }

  for (const [attr, api, summarize] of [
    ["upsert", "upsert", _summarizeUpsert],
    ["update", "update", _summarizeUpdate],
    ["delete", "delete", _summarizeDelete], // Python / legacy name
    ["deleteOne", "delete", _summarizeDelete],
    ["deleteMany", "delete", _summarizeDelete],
    ["deleteAll", "delete_all", _summarizeDeleteAll],
  ] as [string, string, typeof _summarizeUpsert][]) {
    if (typeof proto[attr] === "function") {
      proto[attr] = makeAsyncInvalidatingWrapper(
        proto[attr] as (this: unknown, ...args: unknown[]) => Promise<unknown>,
        TraceType.Pinecone, api, summarize, _pineconeTarget
      );
    }
  }

  if (typeof proto["fetch"] === "function") {
    proto["fetch"] = makeAsyncWrapper(
      proto["fetch"] as (this: unknown, ...args: unknown[]) => Promise<unknown>,
      TraceType.Pinecone, "fetch", _summarizeFetch
    );
  }
}

export function patchPinecone(): void {
  for (const spec of [
    ["@pinecone-database/pinecone", "Index"],
    ["pinecone", "Index"], // older / alternate package name
  ] as [string, string][]) {
    try {
      const mod = require(spec[0]) as Record<string, unknown>;
      const cls = mod[spec[1]] as { prototype: Record<string, unknown> } | undefined;
      if (cls && cls.prototype) {
        _patchIndex(cls);
        return;
      }
    } catch { /* try next */ }
  }
}
