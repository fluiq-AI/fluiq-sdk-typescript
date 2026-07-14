/**
 * Resolve DAG fan-in (join) parents for LangGraph nodes — TS port of the Python
 * SDK's `langgraph_edges`.
 *
 * Modern LangGraph names a node's `langgraph_triggers` after the destination
 * channel (e.g. `branch:to:synthesize`), NOT the source nodes — so trigger
 * parsing alone can't see a fan-in. We instead capture the graph's declared
 * edges at compile time and resolve a join node's parents from its static
 * predecessors. `target -> {sources}`, merged across every compiled graph in
 * the process; join resolution always intersects this with the per-run registry
 * of nodes that actually ran, so cross-graph node-name collisions can never
 * invent a false parent.
 */

import { AsyncLocalStorage } from "node:async_hooks";

// target node name -> set of source node names.
const STATIC_PREDS = new Map<string, Set<string>>();

// ── Current graph name (published around a compiled-graph invocation) ─────────
// LangGraph's top-level Pregel run reaches the callback handler with an empty
// `serialized`, so the graph *container* span has no name of its own and ends up
// with an empty agent identity (invisible in the Agents view). The compile patch
// runs each invocation inside `runWithGraphName` so the handler can name the
// parent-less container chain. AsyncLocalStorage keeps it correct under
// concurrent / async graph runs.
const GRAPH_NAME = new AsyncLocalStorage<string>();

/** Run `fn` with `name` published as the current graph name. */
export function runWithGraphName<T>(name: string, fn: () => T): T {
  if (!name) return fn();
  return GRAPH_NAME.run(name, fn);
}

/** The name of the graph currently being invoked, if any. */
export function currentGraphName(): string | null {
  return GRAPH_NAME.getStore() ?? null;
}

// LangGraph's START / END sentinels — never real join parents.
const EDGE_SENTINELS = new Set(["__start__", "__end__"]);

const TOKEN_RE = /[^A-Za-z0-9_]+/;

/** Split trigger strings into node-name tokens, format-agnostically. */
export function triggerTokens(triggers: unknown): Set<string> {
  if (triggers == null) return new Set();
  const text = Array.isArray(triggers)
    ? triggers.map((t) => String(t)).join(" ")
    : String(triggers);
  return new Set(text.split(TOKEN_RE).filter(Boolean));
}

/**
 * Record static `(source, target)` edges (a `StateGraph.edges` Set of 2-tuples)
 * as `target -> {sources}`. Ignores START/END sentinels and malformed entries.
 * Fail-open.
 */
export function registerGraphEdges(edges: unknown): void {
  try {
    if (!edges) return;
    for (const edge of edges as Iterable<unknown>) {
      if (!Array.isArray(edge) || edge.length < 2) continue;
      const src = String(edge[0]);
      const dst = String(edge[1]);
      if (!src || !dst || EDGE_SENTINELS.has(src) || EDGE_SENTINELS.has(dst)) continue;
      let preds = STATIC_PREDS.get(dst);
      if (!preds) {
        preds = new Set();
        STATIC_PREDS.set(dst, preds);
      }
      preds.add(src);
    }
  } catch {
    /* fail-open */
  }
}

/**
 * Static predecessor node names for `nodeName` (sorted), or `[]`. Stamped onto
 * each LangGraph node's trace so the dashboard can draw the real DAG — both
 * fan-in (join) and fan-out edges. `parent_ids` only carries fan-in joins, so
 * single-predecessor edges (e.g. synthesize -> writer) would otherwise be
 * invisible.
 */
export function predecessorNames(nodeName: string | null | undefined): string[] {
  if (!nodeName) return [];
  const preds = STATIC_PREDS.get(nodeName);
  return preds ? [...preds].sort() : [];
}

/**
 * Join parents from the static edge graph, intersected with nodes that ran.
 * Reliable regardless of the LangGraph trigger encoding. Returns predecessor
 * run_ids only for a genuine fan-in (`>= minParents`); `null` otherwise so the
 * single `parent_id` stands.
 */
export function resolveJoinParentsByEdges(
  registry: Map<string, string>,
  nodeName: string | null | undefined,
  minParents = 2
): string[] | null {
  if (!nodeName || registry.size === 0) return null;
  const preds = STATIC_PREDS.get(nodeName);
  if (!preds) return null;
  const out: string[] = [];
  const seen = new Set<string>();
  for (const name of preds) {
    const runId = registry.get(name);
    if (runId && !seen.has(runId)) {
      out.push(runId);
      seen.add(runId);
    }
  }
  return out.length >= minParents ? out : null;
}

/**
 * Fallback: join parents from trigger names matched against the registry (for
 * frameworks / older versions whose triggers name the source nodes).
 */
export function resolveJoinParents(
  registry: Map<string, string>,
  nodeName: string | null | undefined,
  triggers: unknown,
  minParents = 2
): string[] | null {
  const tokens = triggerTokens(triggers);
  if (tokens.size === 0 || registry.size === 0) return null;
  const out: string[] = [];
  const seen = new Set<string>();
  for (const [name, runId] of registry) {
    if (!name || name === nodeName) continue;
    if (tokens.has(name) && !seen.has(runId)) {
      out.push(runId);
      seen.add(runId);
    }
  }
  return out.length >= minParents ? out : null;
}
