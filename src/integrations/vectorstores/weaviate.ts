/**
 * Weaviate vectorstore integration (weaviate-client v3).
 *
 * Unlike Python's class-per-collection layout, the JS client builds each
 * collection's `query`/`data` objects from two module-level factories:
 *   - collections/query/index.js  default export = QueryManager.use(...) factory
 *     (methods live on the QueryManager prototype: nearVector/nearText/hybrid/
 *      bm25/fetchObjects — all async)
 *   - collections/data/index.js   default export = data(...) factory that returns
 *     a plain object of closures (insert/insertMany/deleteById/deleteMany/
 *      update/replace — all async)
 *
 * There is no exported client class with patchable prototype methods, so we wrap
 * each factory's default export: on every call we stamp the collection name onto
 * the returned object and decorate its methods (own properties that shadow the
 * QueryManager prototype for the query side, direct overwrites for the data
 * side). The factory modules are deep files not exposed by the package `exports`
 * map, so they're loaded by absolute path — which resolves to the very same
 * require-cache entry the collection module already holds.
 */
import { TraceType } from "../shared/models";
import {
  buildMatch,
  captureMatches,
  captureQueryTexts,
  makeAsyncWrapper,
  safeJsonable,
  truncateIds,
  vectorDim,
} from "./utils";

// Non-enumerable stamps we attach to each query/data object so the summarizers
// can recover the collection name (the data object is a bare closure literal
// with no `name` of its own; the query instance keeps it on `this.check.name`).
const COLL = "__fluiq_collection";
const TEN = "__fluiq_tenant";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function _collectionName(instance: unknown): string {
  const inst = instance as Record<string, unknown>;
  if (inst == null) return "";
  return String(
    inst[COLL] ??
    (inst["check"] as Record<string, unknown> | undefined)?.["name"] ??
    inst["name"] ??
    ""
  );
}

function _target(instance: unknown, opts: Record<string, unknown>): Record<string, unknown> {
  const inst = instance as Record<string, unknown> | null;
  return {
    collection: _collectionName(instance) || null,
    tenant: opts["tenant"] ?? inst?.[TEN] ?? null,
  };
}

/**
 * The JS methods are positional — `method(primary, opts, callOpts)`. The shared
 * wrappers peel a trailing plain-object arg off into `kwargs`, so the options
 * bag is either the slot right after the primary args, or (when it was the last
 * argument) `kwargs`. `arity` is the count of leading primary args (1 for
 * nearVector/nearText/hybrid/bm25, 0 for fetchObjects).
 */
function _opts(positional: unknown[], kwargs: Record<string, unknown>, arity: number): Record<string, unknown> {
  const cand = positional[arity];
  if (cand != null && typeof cand === "object" && !Array.isArray(cand)) return cand as Record<string, unknown>;
  return kwargs;
}

/** First positional/data argument, falling back to the peeled-off kwargs. */
function _arg0(positional: unknown[], kwargs: Record<string, unknown>): unknown {
  return positional.length > 0 ? positional[0] : kwargs;
}

function _propertiesText(props: unknown): string | null {
  if (typeof props !== "object" || props === null) return null;
  const p = props as Record<string, unknown>;
  for (const key of ["text", "content", "page_content", "chunk", "body"]) {
    if (typeof p[key] === "string") return p[key] as string;
  }
  return null;
}

function _objectsToMatches(response: unknown): { items: Record<string, unknown>[]; scores: number[] } {
  const r = response as Record<string, unknown> | null;
  if (!r) return { items: [], scores: [] };
  const objs = (Array.isArray(r["objects"]) ? r["objects"] : null) as unknown[] | null;
  if (!objs) return { items: [], scores: [] };
  const items: Record<string, unknown>[] = [];
  const scores: number[] = [];
  for (const o of objs) {
    const oRec = o as Record<string, unknown>;
    const oid = oRec["uuid"] ?? oRec["id"] ?? null;
    const meta = oRec["metadata"] as Record<string, unknown> | null;
    const sc = meta ? (meta["score"] ?? meta["distance"] ?? null) : null;
    if (typeof sc === "number") scores.push(sc);
    const props = oRec["properties"];
    items.push(buildMatch({ id: oid, score: sc as number | undefined, text: _propertiesText(props) ?? undefined, metadata: props }));
  }
  return { items, scores };
}

function _attachMatches(out: Record<string, unknown>, response: unknown): void {
  const { items, scores } = _objectsToMatches(response);
  if (items.length > 0) {
    out["result"] = {
      matches: captureMatches(items),
      score_min: scores.length > 0 ? Math.min(...scores) : null,
      score_max: scores.length > 0 ? Math.max(...scores) : null,
    };
  }
}

function _objVectorDim(obj: Record<string, unknown> | null | undefined): number | null {
  if (obj == null) return null;
  const v = obj["vector"] ?? obj["vectors"];
  if (Array.isArray(v)) return vectorDim(v);
  // Named-vector form: { vectors: { default: number[] } }
  if (v != null && typeof v === "object") {
    const def = (v as Record<string, unknown>)["default"];
    if (Array.isArray(def)) return vectorDim(def);
  }
  return null;
}

// ---------------------------------------------------------------------------
// Query summarizers
// ---------------------------------------------------------------------------

function _summarizeNearVector(
  positional: unknown[], kwargs: Record<string, unknown>, instance: unknown, response?: unknown
): Record<string, unknown> {
  const vec = positional[0] ?? null;
  const opts = _opts(positional, kwargs, 1);
  const out: Record<string, unknown> = {
    target: _target(instance, opts),
    query: {
      top_k: opts["limit"],
      vector_dim: vectorDim(vec),
      filter: safeJsonable(opts["filters"]),
      distance: opts["distance"],
      certainty: opts["certainty"],
      target_vector: opts["targetVector"],
    },
  };
  _attachMatches(out, response);
  return out;
}

function _summarizeNearText(
  positional: unknown[], kwargs: Record<string, unknown>, instance: unknown, response?: unknown
): Record<string, unknown> {
  const q = positional[0] ?? null;
  const opts = _opts(positional, kwargs, 1);
  const out: Record<string, unknown> = {
    target: _target(instance, opts),
    query: {
      top_k: opts["limit"],
      texts: captureQueryTexts(q),
      filter: safeJsonable(opts["filters"]),
      distance: opts["distance"],
      certainty: opts["certainty"],
      target_vector: opts["targetVector"],
    },
  };
  _attachMatches(out, response);
  return out;
}

function _summarizeHybrid(
  positional: unknown[], kwargs: Record<string, unknown>, instance: unknown, response?: unknown
): Record<string, unknown> {
  const q = positional[0] ?? null;
  const opts = _opts(positional, kwargs, 1);
  const out: Record<string, unknown> = {
    target: _target(instance, opts),
    query: {
      top_k: opts["limit"],
      alpha: opts["alpha"],
      texts: captureQueryTexts(q),
      vector_dim: vectorDim(opts["vector"]),
      filter: safeJsonable(opts["filters"]),
    },
  };
  _attachMatches(out, response);
  return out;
}

function _summarizeBm25(
  positional: unknown[], kwargs: Record<string, unknown>, instance: unknown, response?: unknown
): Record<string, unknown> {
  const q = positional[0] ?? null;
  const opts = _opts(positional, kwargs, 1);
  const out: Record<string, unknown> = {
    target: _target(instance, opts),
    query: {
      top_k: opts["limit"],
      texts: captureQueryTexts(q),
      filter: safeJsonable(opts["filters"]),
    },
  };
  _attachMatches(out, response);
  return out;
}

function _summarizeFetchObjects(
  positional: unknown[], kwargs: Record<string, unknown>, instance: unknown, response?: unknown
): Record<string, unknown> {
  const opts = _opts(positional, kwargs, 0);
  const out: Record<string, unknown> = {
    target: _target(instance, opts),
    query: {
      limit: opts["limit"],
      offset: opts["offset"],
      filter: safeJsonable(opts["filters"]),
    },
  };
  _attachMatches(out, response);
  return out;
}

// ---------------------------------------------------------------------------
// Data summarizers
// ---------------------------------------------------------------------------

function _summarizeInsert(
  positional: unknown[], kwargs: Record<string, unknown>, instance: unknown, response?: unknown
): Record<string, unknown> {
  const obj = _arg0(positional, kwargs) as Record<string, unknown> | null;
  const out: Record<string, unknown> = {
    target: _target(instance, kwargs),
    mutation: {
      vector_dim: _objVectorDim(obj),
      has_properties: obj != null,
    },
  };
  // insert() resolves to the new object's id (a string/uuid).
  if (response != null && (typeof response === "string" || typeof response === "number")) {
    (out["mutation"] as Record<string, unknown>)["id"] = String(response);
  }
  return out;
}

function _summarizeInsertMany(
  positional: unknown[], kwargs: Record<string, unknown>, instance: unknown, response?: unknown
): Record<string, unknown> {
  const objects = _arg0(positional, kwargs);
  let count: number | null = null;
  let dim: number | null = null;
  if (objects != null) {
    try {
      const seq = Array.from(objects as Iterable<unknown>);
      count = seq.length;
      for (const o of seq) {
        dim = _objVectorDim(o as Record<string, unknown>);
        if (dim != null) break;
      }
    } catch { /* ignore */ }
  }
  const out: Record<string, unknown> = {
    target: _target(instance, kwargs),
    mutation: { vector_count: count, vector_dim: dim },
  };
  if (response != null && typeof response === "object") {
    const r = response as Record<string, unknown>;
    const hasErrors = r["hasErrors"] ?? r["has_errors"];
    if (hasErrors != null) (out["mutation"] as Record<string, unknown>)["has_errors"] = Boolean(hasErrors);
  }
  return out;
}

function _summarizeDeleteById(
  positional: unknown[], kwargs: Record<string, unknown>, instance: unknown, _response?: unknown
): Record<string, unknown> {
  const id = _arg0(positional, kwargs);
  return {
    target: _target(instance, kwargs),
    mutation: { ids: truncateIds(id != null ? [id] : null) },
  };
}

function _summarizeDeleteMany(
  positional: unknown[], kwargs: Record<string, unknown>, instance: unknown, _response?: unknown
): Record<string, unknown> {
  const where = _arg0(positional, kwargs);
  return {
    target: _target(instance, kwargs),
    mutation: { where: safeJsonable(where) },
  };
}

function _summarizeUpdate(
  positional: unknown[], kwargs: Record<string, unknown>, instance: unknown, _response?: unknown
): Record<string, unknown> {
  const obj = (_arg0(positional, kwargs) as Record<string, unknown> | null) ?? {};
  return {
    target: _target(instance, kwargs),
    mutation: {
      ids: truncateIds(obj["id"] != null ? [obj["id"]] : null),
      has_properties: obj["properties"] != null,
      vector_dim: _objVectorDim(obj),
    },
  };
}

const QUERY_OPS = [
  ["nearVector", "near_vector", _summarizeNearVector],
  ["nearText", "near_text", _summarizeNearText],
  ["hybrid", "hybrid", _summarizeHybrid],
  ["bm25", "bm25", _summarizeBm25],
  ["fetchObjects", "fetch_objects", _summarizeFetchObjects],
] as [string, string, typeof _summarizeNearVector][];

const DATA_OPS = [
  ["insert", "insert", _summarizeInsert],
  ["insertMany", "insert_many", _summarizeInsertMany],
  ["deleteById", "delete_by_id", _summarizeDeleteById],
  ["deleteMany", "delete_many", _summarizeDeleteMany],
  ["update", "update", _summarizeUpdate],
  ["replace", "replace", _summarizeUpdate],
] as [string, string, typeof _summarizeInsert][];

// ---------------------------------------------------------------------------
// Factory wrapping
// ---------------------------------------------------------------------------

function _stamp(obj: Record<string, unknown>, collection: unknown, tenant: unknown): void {
  for (const [key, value] of [[COLL, collection], [TEN, tenant]] as [string, unknown][]) {
    try {
      Object.defineProperty(obj, key, { value: value ?? null, enumerable: false, configurable: true, writable: true });
    } catch { /* ignore */ }
  }
}

function _decorateQuery(obj: Record<string, unknown>): void {
  for (const [attr, api, summarize] of QUERY_OPS) {
    const orig = obj[attr];
    if (typeof orig === "function") {
      obj[attr] = makeAsyncWrapper(
        orig as (this: unknown, ...args: unknown[]) => Promise<unknown>,
        TraceType.Weaviate, api, summarize
      );
    }
  }
}

function _decorateData(obj: Record<string, unknown>): void {
  for (const [attr, api, summarize] of DATA_OPS) {
    const orig = obj[attr];
    if (typeof orig === "function") {
      obj[attr] = makeAsyncWrapper(
        orig as (this: unknown, ...args: unknown[]) => Promise<unknown>,
        TraceType.Weaviate, api, summarize
      );
    }
  }
}

/**
 * Wrap a factory module's default export so every returned collection object is
 * stamped with its name and has its methods decorated. `factoryArgs[1]` is the
 * collection name and `factoryArgs[4]` the tenant for both factories
 * (connection, name, dbVersionSupport, consistencyLevel, tenant).
 */
function _wrapFactory(mod: Record<string, unknown>, decorate: (obj: Record<string, unknown>) => void): void {
  if (typeof mod["default"] !== "function" || mod["__fluiq_patched"]) return;
  const origFactory = mod["default"] as (this: unknown, ...args: unknown[]) => unknown;
  const patched = function (this: unknown, ...factoryArgs: unknown[]): unknown {
    const obj = origFactory.apply(this, factoryArgs);
    try {
      if (obj != null && typeof obj === "object") {
        _stamp(obj as Record<string, unknown>, factoryArgs[1], factoryArgs[4]);
        decorate(obj as Record<string, unknown>);
      }
    } catch { /* fail open — never break the client */ }
    return obj;
  };
  mod["default"] = patched;
  mod["__fluiq_patched"] = true;
}

export function patchWeaviate(): void {
  const path = require("path") as typeof import("path");

  let mainPath: string | undefined;
  for (const pkg of ["weaviate-client", "weaviate-ts-client"]) {
    try {
      mainPath = require.resolve(pkg);
      break;
    } catch { /* try next */ }
  }
  if (!mainPath) return; // weaviate not installed

  // The query/data factory modules are deep files the package `exports` map does
  // not expose, so load them by absolute path (same require-cache entry the
  // collection module holds). Fall back from an esm-resolved main to cjs.
  const dirs = [path.dirname(mainPath)];
  const cjsDir = path.dirname(mainPath).replace(/([\\/])esm([\\/])/, "$1cjs$2");
  if (cjsDir !== dirs[0]) dirs.push(cjsDir);

  const load = (rel: string): Record<string, unknown> | undefined => {
    for (const d of dirs) {
      try {
        return require(path.join(d, rel)) as Record<string, unknown>;
      } catch { /* try next dir */ }
    }
    return undefined;
  };

  const queryMod = load(path.join("collections", "query", "index.js"));
  if (queryMod) _wrapFactory(queryMod, _decorateQuery);

  const dataMod = load(path.join("collections", "data", "index.js"));
  if (dataMod) _wrapFactory(dataMod, _decorateData);
}
