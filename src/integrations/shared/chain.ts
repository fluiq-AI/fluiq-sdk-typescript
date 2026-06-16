import { createHash } from "crypto";
import { randomUUID } from "crypto";
import { AsyncLocalStorage } from "async_hooks";

const CONVERSATION_KEYS = ["contents", "messages", "input"] as const;

interface ChainState {
  id: string;
  hashes: string[];
}

const _chainState = new AsyncLocalStorage<{ state: ChainState | null }>();

function _getChainStateRef(): { state: ChainState | null } {
  let ref = _chainState.getStore();
  if (!ref) {
    // No active chain context — create a module-level fallback
    ref = { state: null };
    // We can't run into the storage without a callback, so we use a module-level ref
  }
  return ref;
}

// Module-level fallback for when there's no AsyncLocalStorage context
let _moduleChainState: ChainState | null = null;

function _getState(): ChainState | null {
  const store = _chainState.getStore();
  if (store) return store.state;
  return _moduleChainState;
}

function _setState(s: ChainState | null): void {
  const store = _chainState.getStore();
  if (store) {
    store.state = s;
  } else {
    _moduleChainState = s;
  }
}

function _extractTurns(data: Record<string, unknown>): unknown[] | null {
  for (const key of CONVERSATION_KEYS) {
    const value = data[key];
    if (!Array.isArray(value) || value.length === 0) continue;
    // Flatten one level if it's a list-of-lists (LangChain pattern)
    if (value.every((item) => Array.isArray(item))) {
      const flat: unknown[] = [];
      for (const inner of value) flat.push(...(inner as unknown[]));
      return flat.length > 0 ? flat : null;
    }
    return value;
  }
  return null;
}

function _hashTurn(turn: unknown): string {
  let canonical: string;
  try {
    canonical = JSON.stringify(_normalize(turn), null, 0);
  } catch {
    canonical = String(turn);
  }
  return createHash("sha1").update(canonical, "utf8").digest("hex");
}

function _normalize(v: unknown): unknown {
  if (v === null || v === undefined) return null;
  if (typeof v === "string" || typeof v === "number" || typeof v === "boolean") return v;
  if (Array.isArray(v)) return v.map(_normalize);
  if (typeof v === "object") {
    const obj = v as Record<string, unknown>;
    return Object.fromEntries(
      Object.keys(obj)
        .sort()
        .map((k) => [k, _normalize(obj[k])])
    );
  }
  return String(v);
}

export function computeChainId(data: Record<string, unknown>): string | null {
  const turns = _extractTurns(data);
  const state = _getState();

  if (turns == null) {
    // Non-LLM trace — inherit active chain if any
    if (state != null) return state.id;
    return null;
  }

  const hashes = turns.map(_hashTurn);

  if (state != null) {
    const { id: prevId, hashes: prevHashes } = state;
    if (
      hashes.length >= prevHashes.length &&
      hashes.slice(0, prevHashes.length).join(",") === prevHashes.join(",")
    ) {
      _setState({ id: prevId, hashes });
      return prevId;
    }
  }

  const chainId = randomUUID();
  _setState({ id: chainId, hashes });
  return chainId;
}

export function resetChainState(): void {
  _setState(null);
}
