/**
 * FAISS vectorstore integration.
 * Patches all faiss.Index subclasses' search (cached), add/add_with_ids/
 * remove_ids/train/reset (invalidating), range_search (plain).
 *
 * Note: faiss-node / faiss-js are limited in the JS ecosystem; this patch
 * is applied defensively and will be a no-op if faiss is not installed.
 */
import { TraceType } from "../shared/models";
import {
  truncateIds,
  makeSyncCachedWrapper,
  makeSyncInvalidatingWrapper,
  makeSyncWrapper,
} from "./utils";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// faiss-node exposes dim/ntotal/is_trained as accessor METHODS
// (getDimension(), ntotal(), isTrained()); Python faiss uses plain attributes
// (d, ntotal, is_trained). Try the method first, fall back to the property.
function _accessor(inst: Record<string, unknown>, method: string, prop: string): unknown {
  const fn = inst[method];
  if (typeof fn === "function") {
    try {
      return (fn as () => unknown).call(inst);
    } catch {
      /* fall through */
    }
  }
  return inst[prop] ?? null;
}

function _indexType(inst: Record<string, unknown>): string {
  return (inst.constructor as { name?: string })?.name ?? "Index";
}

function _faissTarget(
  _args: unknown[],
  _kwargs: Record<string, unknown>,
  instance: unknown
): string {
  const inst = instance as Record<string, unknown>;
  return `${_indexType(inst)}:${_accessor(inst, "getDimension", "d") ?? ""}`;
}

function _target(instance: unknown): Record<string, unknown> {
  const inst = instance as Record<string, unknown>;
  return {
    index_type: _indexType(inst),
    dim: _accessor(inst, "getDimension", "d"),
    ntotal: _accessor(inst, "ntotal", "ntotal"),
    is_trained: _accessor(inst, "isTrained", "is_trained"),
  };
}

function _nDFromArg(x: unknown): { n: number | null; d: number | null } {
  if (x == null) return { n: null, d: null };
  // Typed arrays / ndarray-style objects
  const xRec = x as Record<string, unknown>;
  const shape = xRec["shape"];
  if (Array.isArray(shape)) {
    if (shape.length === 1) return { n: 1, d: Number(shape[0]) };
    if (shape.length >= 2) return { n: Number(shape[0]), d: Number(shape[1]) };
  }
  // Plain array: a batch [[...],[...]] vs a single flat vector [..floats..].
  // faiss-node passes a single flat number[] to add()/search().
  if (Array.isArray(x)) {
    if (x.length > 0 && Array.isArray(x[0])) return { n: x.length, d: (x[0] as unknown[]).length };
    return { n: 1, d: x.length };
  }
  return { n: null, d: null };
}

function _flatMinMax(arr: unknown): { min: number | null; max: number | null } {
  if (arr == null) return { min: null, max: null };
  try {
    let flat: number[];
    const arrRec = arr as Record<string, unknown>;
    if (typeof arrRec["reshape"] === "function") {
      flat = Array.from(arrRec["reshape"](-1) as Iterable<number>);
    } else {
      flat = Array.isArray(arr) ? (arr as number[]).flat(Infinity) : [];
    }
    flat = flat.filter((v) => typeof v === "number" && isFinite(v));
    if (flat.length === 0) return { min: null, max: null };
    return { min: Math.min(...flat), max: Math.max(...flat) };
  } catch {
    return { min: null, max: null };
  }
}

function _summarizeAdd(
  args: unknown[],
  kwargs: Record<string, unknown>,
  instance: unknown,
  _response?: unknown
): Record<string, unknown> {
  const x = kwargs["x"] ?? (args[0] ?? null);
  const { n, d } = _nDFromArg(x);
  return { target: _target(instance), mutation: { vector_count: n, vector_dim: d } };
}

function _summarizeAddWithIds(
  args: unknown[],
  kwargs: Record<string, unknown>,
  instance: unknown,
  _response?: unknown
): Record<string, unknown> {
  const x = kwargs["x"] ?? (args[0] ?? null);
  const ids = kwargs["ids"] ?? (args.length > 1 ? args[1] : null);
  const { n, d } = _nDFromArg(x);
  let idSummary = null;
  try {
    const seq = ids != null ? Array.from(ids as Iterable<unknown>) : null;
    idSummary = seq ? truncateIds(seq) : null;
  } catch { /* ignore */ }
  return { target: _target(instance), mutation: { vector_count: n, vector_dim: d, ids: idSummary } };
}

function _summarizeSearch(
  args: unknown[],
  kwargs: Record<string, unknown>,
  instance: unknown,
  response?: unknown
): Record<string, unknown> {
  const x = kwargs["x"] ?? (args[0] ?? null);
  const k = kwargs["k"] ?? (args.length > 1 ? args[1] : null);
  const { n, d } = _nDFromArg(x);
  const out: Record<string, unknown> = {
    target: _target(instance),
    query: { top_k: k, vector_count: n, vector_dim: d },
  };
  // faiss-node returns { distances, labels }; older bindings return [D, I].
  if (response && typeof response === "object" && !Array.isArray(response) && "labels" in (response as object)) {
    const r = response as Record<string, unknown>;
    const labels = r["labels"];
    const { min: dMin, max: dMax } = _flatMinMax(r["distances"]);
    out["result"] = {
      count: Array.isArray(labels) ? labels.length : null,
      labels: Array.isArray(labels) ? labels.slice(0, 10) : null,
      distance_min: dMin,
      distance_max: dMax,
    };
  } else if (Array.isArray(response) && response.length >= 2) {
    const D = response[0];
    const I = response[1];
    const { min: dMin, max: dMax } = _flatMinMax(D);
    let resultCount: number | null = null;
    try {
      const iRec = I as Record<string, unknown>;
      resultCount = typeof iRec["size"] === "number" ? iRec["size"] : null;
    } catch { /* ignore */ }
    out["result"] = { count: resultCount, distance_min: dMin, distance_max: dMax };
  }
  return out;
}

function _summarizeRangeSearch(
  args: unknown[],
  kwargs: Record<string, unknown>,
  instance: unknown,
  response?: unknown
): Record<string, unknown> {
  const x = kwargs["x"] ?? (args[0] ?? null);
  const radius = kwargs["radius"] ?? (args.length > 1 ? args[1] : null);
  const { n, d } = _nDFromArg(x);
  const out: Record<string, unknown> = {
    target: _target(instance),
    query: {
      vector_count: n,
      vector_dim: d,
      radius: typeof radius === "number" ? radius : null,
    },
  };
  if (Array.isArray(response) && response.length >= 3) {
    const lims = response[0];
    const D = response[1];
    let total: number | null = null;
    try {
      const limsArr = Array.isArray(lims) ? lims : [];
      if (limsArr.length > 0) total = Number(limsArr[limsArr.length - 1]);
    } catch { /* ignore */ }
    const { min: dMin, max: dMax } = _flatMinMax(D);
    out["result"] = { count: total, distance_min: dMin, distance_max: dMax };
  }
  return out;
}

function _summarizeRemoveIds(
  args: unknown[],
  kwargs: Record<string, unknown>,
  instance: unknown,
  response?: unknown
): Record<string, unknown> {
  const sel = kwargs["sel"] ?? (args[0] ?? null);
  const out: Record<string, unknown> = {
    target: _target(instance),
    mutation: { selector: sel != null ? (sel as object).constructor.name : null },
  };
  if (typeof response === "number") (out["mutation"] as Record<string, unknown>)["removed_count"] = response;
  return out;
}

function _summarizeTrain(
  args: unknown[], kwargs: Record<string, unknown>, instance: unknown, _response?: unknown
): Record<string, unknown> {
  const x = kwargs["x"] ?? (args[0] ?? null);
  const { n, d } = _nDFromArg(x);
  return { target: _target(instance), mutation: { vector_count: n, vector_dim: d } };
}

function _summarizeReset(
  _args: unknown[], _kwargs: Record<string, unknown>, instance: unknown, _response?: unknown
): Record<string, unknown> {
  return { target: _target(instance), mutation: {} };
}

async function _searchCacheKey(
  args: unknown[],
  kwargs: Record<string, unknown>,
  instance: unknown
): Promise<string> {
  const x = kwargs["x"] ?? (args[0] ?? null);
  const k = kwargs["k"] ?? (args.length > 1 ? args[1] : null);
  const inst = instance as Record<string, unknown>;
  const idxType = (inst.constructor as { name?: string }).name ?? "Index";
  const dim = inst["d"] ?? null;
  let qList: unknown;
  try {
    qList = typeof (x as Record<string, unknown>)["tolist"] === "function"
      ? ((x as Record<string, unknown>)["tolist"] as () => unknown)()
      : Array.from(x as Iterable<unknown>);
  } catch {
    qList = String(x);
  }
  const { vectorstoreCacheKey } = require("../../optimization/client") as typeof import("../../optimization/client");
  return vectorstoreCacheKey("faiss", `${idxType}:${dim}`, qList, k, null);
}

function _searchRawResult(
  _args: unknown[],
  _kwargs: Record<string, unknown>,
  _instance: unknown,
  response: unknown
): Record<string, unknown> | null {
  // faiss-node: { distances, labels }
  if (response && typeof response === "object" && !Array.isArray(response) && "labels" in (response as object)) {
    const r = response as Record<string, unknown>;
    return { distances: r["distances"] ?? null, labels: r["labels"] ?? null };
  }
  // Legacy bindings: [D, I]
  if (!Array.isArray(response) || response.length < 2) return null;
  const D = response[0];
  const I = response[1];
  try {
    return {
      distances: typeof (D as Record<string, unknown>)["tolist"] === "function"
        ? ((D as Record<string, unknown>)["tolist"] as () => unknown)()
        : Array.from(D as Iterable<unknown>),
      indices: typeof (I as Record<string, unknown>)["tolist"] === "function"
        ? ((I as Record<string, unknown>)["tolist"] as () => unknown)()
        : Array.from(I as Iterable<unknown>),
    };
  } catch {
    return null;
  }
}

function _searchMock(
  cachedResult: Record<string, unknown>,
  _args: unknown[],
  _kwargs: Record<string, unknown>,
  _instance: unknown
): unknown {
  // Reconstruct faiss-node's { distances, labels } return shape from the cache.
  return {
    distances: cachedResult["distances"] ?? [],
    labels: cachedResult["labels"] ?? cachedResult["indices"] ?? [],
  };
}

// ---------------------------------------------------------------------------
// Instance decoration
// ---------------------------------------------------------------------------
//
// faiss-node is a native (NAPI) addon: its prototype methods are
// non-writable / non-configurable, so the methods can't be wrapped on the
// prototype (assignment silently no-ops; defineProperty throws). Instances,
// however, can be given own properties via defineProperty, which shadow the
// locked prototype methods. So we wrap the *constructor* on the module exports
// and decorate each instance as it is created.

const _INVALIDATING: [string, string, typeof _summarizeAdd][] = [
  ["add", "add", _summarizeAdd],
  ["add_with_ids", "add_with_ids", _summarizeAddWithIds],
  ["remove_ids", "remove_ids", _summarizeRemoveIds],
  ["removeIds", "remove_ids", _summarizeRemoveIds], // faiss-node camelCase
  ["mergeFrom", "merge_from", _summarizeReset], // faiss-node: merge another index
  ["train", "train", _summarizeTrain],
  ["reset", "reset", _summarizeReset],
];

function _define(inst: Record<string, unknown>, name: string, fn: unknown): void {
  try {
    Object.defineProperty(inst, name, { value: fn, writable: true, configurable: true, enumerable: false });
  } catch {
    // instance already has a locked own prop (unexpected) — skip
  }
}

/** Installs traced/cached method wrappers as own properties on a fresh index instance. */
function _decorateInstance(inst: Record<string, unknown>, proto: Record<string, unknown>): void {
  if (typeof proto["search"] === "function") {
    _define(inst, "search", makeSyncCachedWrapper(
      proto["search"] as (this: unknown, ...args: unknown[]) => unknown,
      TraceType.FAISS, "search", _summarizeSearch, _searchCacheKey, _searchMock, _searchRawResult
    ));
  }
  const definedApis = new Set<string>();
  for (const [attr, api, summarize] of _INVALIDATING) {
    if (typeof proto[attr] === "function" && !definedApis.has(attr)) {
      definedApis.add(attr);
      _define(inst, attr, makeSyncInvalidatingWrapper(
        proto[attr] as (this: unknown, ...args: unknown[]) => unknown,
        TraceType.FAISS, api, summarize, _faissTarget
      ));
    }
  }
  if (typeof proto["range_search"] === "function") {
    _define(inst, "range_search", makeSyncWrapper(
      proto["range_search"] as (this: unknown, ...args: unknown[]) => unknown,
      TraceType.FAISS, "range_search", _summarizeRangeSearch
    ));
  }
}

export function patchFAISS(): void {
  let faiss: Record<string, unknown>;
  try {
    faiss = require("faiss-node") as Record<string, unknown>;
  } catch {
    try {
      faiss = require("faiss") as Record<string, unknown>;
    } catch {
      return; // faiss not installed
    }
  }

  for (const name of Object.keys(faiss)) {
    try {
      const Orig = faiss[name] as
        | ((new (...args: unknown[]) => unknown) & { prototype?: Record<string, unknown>; _fluiqWrapped?: boolean })
        | undefined;
      // An index class is any constructor whose prototype exposes `search`.
      if (typeof Orig !== "function" || !Orig.prototype || typeof Orig.prototype["search"] !== "function") continue;
      if (Orig._fluiqWrapped) continue;

      const proto = Orig.prototype;
      const Wrapped = function (this: unknown, ...args: unknown[]): unknown {
        const inst = new (Orig as new (...a: unknown[]) => Record<string, unknown>)(...args);
        _decorateInstance(inst, proto);
        return inst;
      } as unknown as ((new (...args: unknown[]) => unknown)) & { _fluiqWrapped?: boolean; prototype: unknown };

      Wrapped.prototype = proto; // keep instanceof working
      Object.setPrototypeOf(Wrapped, Orig); // inherit static methods (read/fromBuffer/…)
      Wrapped._fluiqWrapped = true;

      try {
        faiss[name] = Wrapped;
      } catch {
        // export not writable on this binding — nothing more we can do
      }
    } catch {
      /* ignore */
    }
  }
}
