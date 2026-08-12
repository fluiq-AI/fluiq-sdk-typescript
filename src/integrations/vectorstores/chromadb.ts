/**
 * ChromaDB vectorstore integration (chromadb JS v3).
 * Patches CollectionHandle.query (cached), .add/upsert/update/delete
 * (invalidating), .get/count (plain). The JS client's methods are all async and
 * take a single camelCase options object (queryTexts/queryEmbeddings/nResults/
 * whereDocument/...), unlike Python's snake_case keyword args — both spellings
 * are accepted below for cross-SDK parity.
 */
import { TraceType } from "../shared/models";
import {
  buildMatch,
  captureMatches,
  captureQueryTexts,
  safeJsonable,
  truncateIds,
  vectorCountAndDim,
  makeAsyncWrapper,
} from "./utils";

// ---------------------------------------------------------------------------
// Target / summary helpers
// ---------------------------------------------------------------------------

function _target(instance: unknown): Record<string, unknown> {
  return { collection: (instance as Record<string, unknown>)["name"] ?? null };
}

function _summarizeQuery(
  _args: unknown[],
  kwargs: Record<string, unknown>,
  instance: unknown,
  response?: unknown
): Record<string, unknown> {
  const qe = kwargs["queryEmbeddings"] ?? kwargs["query_embeddings"];
  const qt = kwargs["queryTexts"] ?? kwargs["query_texts"];
  const vc = vectorCountAndDim(qe);
  const out: Record<string, unknown> = {
    target: _target(instance),
    query: {
      top_k: kwargs["nResults"] ?? kwargs["n_results"],
      vector_count: vc.count,
      vector_dim: vc.dim,
      texts: captureQueryTexts(qt),
      filter: safeJsonable(kwargs["where"]),
      where_document: safeJsonable(kwargs["whereDocument"] ?? kwargs["where_document"]),
      include: kwargs["include"],
    },
  };

  if (response != null && typeof response === "object") {
    const r = response as Record<string, unknown>;
    const ids = (r["ids"] as unknown[][]) ?? [];
    const distances = (r["distances"] as unknown[][]) ?? [];
    const documents = (r["documents"] as unknown[][]) ?? [];
    const metadatas = (r["metadatas"] as unknown[][]) ?? [];
    const matches: Record<string, unknown>[] = [];
    const flatD: number[] = [];

    for (let qi = 0; qi < ids.length; qi++) {
      const subIds = Array.isArray(ids[qi]) ? ids[qi] : [];
      const subD = Array.isArray(distances[qi]) ? distances[qi] : [];
      const subDoc = Array.isArray(documents[qi]) ? documents[qi] : [];
      const subMd = Array.isArray(metadatas[qi]) ? metadatas[qi] : [];
      for (let j = 0; j < subIds.length; j++) {
        const dist = typeof subD[j] === "number" ? subD[j] as number : null;
        if (dist != null) flatD.push(dist);
        matches.push(buildMatch({
          id: subIds[j],
          score: dist ?? undefined,
          text: subDoc[j] as string | undefined,
          metadata: subMd[j],
        }));
      }
    }
    out["result"] = {
      matches: captureMatches(matches),
      distance_min: flatD.length > 0 ? Math.min(...flatD) : null,
      distance_max: flatD.length > 0 ? Math.max(...flatD) : null,
    };
  }
  return out;
}

function _summarizeMutation(
  _args: unknown[],
  kwargs: Record<string, unknown>,
  instance: unknown,
  _response?: unknown
): Record<string, unknown> {
  const ids = kwargs["ids"];
  const embeddings = kwargs["embeddings"];
  const documents = kwargs["documents"];
  const vc = vectorCountAndDim(embeddings);
  return {
    target: _target(instance),
    mutation: {
      ids: truncateIds(ids),
      vector_count: vc.count,
      vector_dim: vc.dim,
      document_count: Array.isArray(documents) ? documents.length : null,
    },
  };
}

function _summarizeGet(
  _args: unknown[],
  kwargs: Record<string, unknown>,
  instance: unknown,
  response?: unknown
): Record<string, unknown> {
  const out: Record<string, unknown> = {
    target: _target(instance),
    query: {
      ids: truncateIds(kwargs["ids"]),
      filter: safeJsonable(kwargs["where"]),
      where_document: safeJsonable(kwargs["whereDocument"] ?? kwargs["where_document"]),
      limit: kwargs["limit"],
      offset: kwargs["offset"],
    },
  };
  if (response != null && typeof response === "object") {
    const r = response as Record<string, unknown>;
    out["result"] = { ids: truncateIds(r["ids"]) };
  }
  return out;
}

function _summarizeCount(
  _args: unknown[],
  _kwargs: Record<string, unknown>,
  instance: unknown,
  response?: unknown
): Record<string, unknown> {
  const out: Record<string, unknown> = { target: _target(instance) };
  if (typeof response === "number") out["result"] = { count: response };
  return out;
}

function _summarizeDelete(
  _args: unknown[],
  kwargs: Record<string, unknown>,
  instance: unknown,
  _response?: unknown
): Record<string, unknown> {
  return {
    target: _target(instance),
    mutation: {
      ids: truncateIds(kwargs["ids"]),
      filter: safeJsonable(kwargs["where"]),
      where_document: safeJsonable(kwargs["whereDocument"] ?? kwargs["where_document"]),
    },
  };
}

// op walks the prototype chain (`count`/`delete`/`get` are inherited from a base
// class), so `in` is used rather than `hasOwnProperty`.
function _patchCollection(Collection: { prototype: Record<string, unknown> }): void {
  const proto = Collection.prototype;

  if ("query" in proto) {
    proto["query"] = makeAsyncWrapper(
      proto["query"] as (this: unknown, ...args: unknown[]) => Promise<unknown>,
      TraceType.ChromaDB, "query", _summarizeQuery
    );
  }

  for (const [attr, api, summarize] of [
    ["add", "add", _summarizeMutation],
    ["upsert", "upsert", _summarizeMutation],
    ["update", "update", _summarizeMutation],
    ["delete", "delete", _summarizeDelete],
  ] as [string, string, typeof _summarizeMutation][]) {
    if (attr in proto) {
      proto[attr] = makeAsyncWrapper(
        proto[attr] as (this: unknown, ...args: unknown[]) => Promise<unknown>,
        TraceType.ChromaDB, api, summarize
      );
    }
  }

  for (const [attr, api, summarize] of [
    ["get", "get", _summarizeGet],
    ["count", "count", _summarizeCount],
  ] as [string, string, typeof _summarizeGet][]) {
    if (attr in proto) {
      proto[attr] = makeAsyncWrapper(
        proto[attr] as (this: unknown, ...args: unknown[]) => Promise<unknown>,
        TraceType.ChromaDB, api, summarize
      );
    }
  }
}

// The collection objects `ChromaClient` hands back are instances of an internal,
// unexported class (`_CollectionImpl`) — the exported `CollectionHandle` is only
// a public type, so patching its prototype never touches a real collection.
// Instead we wrap the client's collection-getter methods and patch the actual
// prototype of whatever they return, exactly once per prototype.
const _patchedProtos = new WeakSet<object>();

function _ensureCollectionPatched(col: unknown): void {
  if (col == null || typeof col !== "object") return;
  const proto = Object.getPrototypeOf(col);
  if (!proto || _patchedProtos.has(proto)) return;
  if (!("query" in proto)) return; // not a collection — ignore
  _patchedProtos.add(proto);
  _patchCollection({ prototype: proto as Record<string, unknown> });
}

function _patchFromResult(result: unknown): void {
  if (Array.isArray(result)) result.forEach(_ensureCollectionPatched);
  else _ensureCollectionPatched(result);
}

// Wrap a client getter so the collection(s) it returns get their prototype
// patched. Preserves sync vs async behaviour (only awaits real thenables) so a
// synchronous getter isn't accidentally turned into a Promise.
function _wrapClientGetter(proto: Record<string, unknown>, name: string): void {
  const orig = proto[name];
  if (typeof orig !== "function") return;
  const fn = orig as (this: unknown, ...args: unknown[]) => unknown;
  const wrappedGetter = function (this: unknown, ...args: unknown[]): unknown {
    const result = fn.apply(this, args);
    if (result != null && typeof (result as { then?: unknown }).then === "function") {
      return (result as Promise<unknown>).then((col) => {
        try { _patchFromResult(col); } catch { /* fail open */ }
        return col;
      });
    }
    try { _patchFromResult(result); } catch { /* fail open */ }
    return result;
  };
  // Marker so a wrapping-regression guard can detect this without relying on the
  // (computed-assignment) function name.
  Object.defineProperty(wrappedGetter, "__fluiqWrapped", { value: true, enumerable: false });
  proto[name] = wrappedGetter;
}

export function patchChromaDB(): void {
  try {
    const mod = require("chromadb") as Record<string, unknown>;
    // CloudClient extends ChromaClient, so patching ChromaClient.prototype covers
    // its inherited getters too; patch any client that defines its own.
    for (const clientName of ["ChromaClient", "CloudClient"]) {
      const cls = mod[clientName] as { prototype: Record<string, unknown> } | undefined;
      const proto = cls?.prototype;
      if (!proto) continue;
      for (const name of [
        "getOrCreateCollection",
        "createCollection",
        "getCollection",
        "getCollectionById",
        "getCollectionByCrn",
        "getCollections",
        "collection",
      ]) {
        if (Object.prototype.hasOwnProperty.call(proto, name)) _wrapClientGetter(proto, name);
      }
    }
  } catch { /* chromadb not installed */ }
}
