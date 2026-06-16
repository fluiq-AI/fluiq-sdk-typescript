/**
 * Optimization client — trace-analysis-driven Redis caching.
 *
 * On the first LLM call the client lazily fetches the cache profile from
 * /optimize/profile. The profile is provisioned by Fluiq's backend after
 * analysing the account's historical traces and contains:
 *
 *   { redis_url, key_prefix, models, ttl_seconds }
 *
 * An empty `models` list means "cache all models".
 */
import axios from "axios";
import { makeKey } from "./caching/base";
import { _state } from "./state";

let _initPromise: Promise<void> | null = null;

async function _ensureInitialized(): Promise<void> {
  if (_state.initialized) return;
  if (_initPromise) return _initPromise;
  _initPromise = _fetchProfile().finally(() => {
    _state.initialized = true;
  });
  return _initPromise;
}

async function _fetchProfile(): Promise<void> {
  const { _config } = require("../config") as typeof import("../config");
  const url = `${_config.endpoint}/${_config.version}/optimize/profile`;
  try {
    const resp = await axios.get(url, {
      headers: { "x-api-key": _config.api_key ?? "" },
      timeout: 5_000,
      validateStatus: () => true,
    });
    if (resp.status === 200) {
      const profile = resp.data as {
        redis_url?: string;
        key_prefix?: string;
        models?: string[];
        ttl_seconds?: number;
      };
      _state.profile = profile;
      const redisUrl = profile.redis_url;
      if (redisUrl) {
        try {
          const { RedisCache } = require("./caching/redisCache") as typeof import("./caching/redisCache");
          _state.cache = new RedisCache(
            redisUrl,
            profile.ttl_seconds,
            profile.key_prefix ?? "fluiq:"
          );
        } catch {
          // ioredis not installed — Redis caching stays disabled (fail open).
        }
      }
    }
  } catch {
    // Profile fetch failed — caching stays disabled (fail open).
  }
}

// ---------------------------------------------------------------------------
// Key helpers
// ---------------------------------------------------------------------------

function _messagesFrom(d: Record<string, unknown>): unknown {
  return d["messages"] ?? d["contents"] ?? d["input"] ?? "";
}

function _cacheKey(d: Record<string, unknown>): string {
  return makeKey(
    "llm",
    d["model"] ?? "",
    _messagesFrom(d),
    d["system"] ?? d["system_instruction"] ?? "",
    d["tools"] ?? d["tool_config"] ?? null,
    d["mcp_servers"] ?? null
  );
}

function _embeddingCacheKey(d: Record<string, unknown>): string {
  return makeKey(
    "embedding",
    d["model"] ?? "",
    d["input"] ?? d["prompt"] ?? d["contents"] ?? d["texts"] ?? ""
  );
}

function _functionCacheKey(funcName: string, argsStr: string): string {
  return makeKey("function", funcName, argsStr);
}

function _toolCacheKey(toolName: string, argsJson: string): string {
  return makeKey("tool", toolName, argsJson);
}

/** Serialize tool args deterministically (object keys sorted) for cache keying. */
function _normalizeArgs(args: Record<string, unknown> | string): string {
  if (typeof args === "object" && args !== null) {
    return JSON.stringify(_sortObject(args));
  }
  try {
    return JSON.stringify(_sortObject(JSON.parse(String(args))));
  } catch {
    return String(args);
  }
}

function _sortObject(v: unknown): unknown {
  if (Array.isArray(v)) return v.map(_sortObject);
  if (v && typeof v === "object") {
    const obj = v as Record<string, unknown>;
    return Object.fromEntries(
      Object.keys(obj)
        .sort()
        .map((k) => [k, _sortObject(obj[k])])
    );
  }
  return v;
}

function _modelAllowed(model: string): boolean {
  const allowed: string[] = (_state.profile?.models) ?? [];
  return allowed.length === 0 || allowed.includes(model);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export async function lookupCache(
  params: Record<string, unknown>
): Promise<Record<string, unknown> | null> {
  await _ensureInitialized();
  if (_state.cache == null) return null;
  if (!_modelAllowed(String(params["model"] ?? ""))) return null;

  const raw = await _state.cache.get(_cacheKey(params));
  if (raw == null) return null;
  if (typeof raw === "string") return { type: "llm", response: raw };
  if (typeof raw === "object") return raw as Record<string, unknown>;
  return null;
}

export async function lookupEmbeddingCache(
  params: Record<string, unknown>
): Promise<Record<string, unknown> | null> {
  await _ensureInitialized();
  if (_state.cache == null) return null;
  if (!_modelAllowed(String(params["model"] ?? ""))) return null;

  const raw = await _state.cache.get(_embeddingCacheKey(params));
  if (raw == null) return null;
  if (typeof raw === "object") return raw as Record<string, unknown>;
  return null;
}

export async function lookupFunctionCache(
  funcName: string,
  argsStr: string
): Promise<Record<string, unknown> | null> {
  await _ensureInitialized();
  if (_state.cache == null) return null;

  const raw = await _state.cache.get(_functionCacheKey(funcName, argsStr));
  if (
    raw != null &&
    typeof raw === "object" &&
    (raw as Record<string, unknown>)["type"] === "function"
  ) {
    return raw as Record<string, unknown>;
  }
  return null;
}

export async function populateFunctionCache(
  funcName: string,
  argsStr: string,
  result: unknown
): Promise<void> {
  if (_state.cache == null) return;
  await _state.cache.set(_functionCacheKey(funcName, argsStr), {
    type: "function",
    result,
  });
}

/**
 * Return the cached result for a tool call, or `null` on miss.
 *
 * `args` can be an object (recommended) or a JSON string. Keys are sorted
 * before hashing so `{ b: 2, a: 1 }` and `{ a: 1, b: 2 }` hash identically.
 */
export async function lookupToolCache(
  toolName: string,
  args: Record<string, unknown> | string
): Promise<unknown> {
  await _ensureInitialized();
  if (_state.cache == null) return null;
  const raw = await _state.cache.get(_toolCacheKey(toolName, _normalizeArgs(args)));
  if (
    raw != null &&
    typeof raw === "object" &&
    (raw as Record<string, unknown>)["type"] === "tool"
  ) {
    return (raw as Record<string, unknown>)["result"];
  }
  return null;
}

/** Store a tool call result in the cache keyed by (toolName, args). */
export async function populateToolCache(
  toolName: string,
  args: Record<string, unknown> | string,
  result: unknown
): Promise<void> {
  if (_state.cache == null) return;
  await _state.cache.set(_toolCacheKey(toolName, _normalizeArgs(args)), {
    type: "tool",
    result,
  });
}

export function populateCache(data: Record<string, unknown>): void {
  if (_state.cache == null) return;
  const model = String(data["model"] ?? "");

  if (data["api"] === "embeddings") {
    const embResp = data["response"];
    if (!embResp) return;
    if (!_modelAllowed(model)) return;
    _state.cache.set(_embeddingCacheKey(data), { type: "embedding", response: embResp });
    return;
  }

  // LLM response
  const responseText = data["response"] ?? null;
  const toolCalls = data["tool_calls"] ?? null;
  const toolUses = data["tool_uses"] ?? null;
  const functionCalls = data["function_calls"] ?? null;
  const mcpCalls = data["mcp_calls"] ?? null;

  if (!responseText && !toolCalls && !toolUses && !functionCalls && !mcpCalls) return;
  if (!_modelAllowed(model)) return;

  _state.cache.set(_cacheKey(data), {
    type: "llm",
    response: responseText,
    tool_calls: toolCalls,
    tool_uses: toolUses,
    function_calls: functionCalls,
    mcp_calls: mcpCalls,
    mcp_results: data["mcp_results"] ?? null,
    mcp_servers: data["mcp_servers"] ?? null,
  });
}

// ---------------------------------------------------------------------------
// Vectorstore cache (generation-based invalidation)
// ---------------------------------------------------------------------------

function _collectionGenKey(integration: string, target: string): string {
  return `_vs_gen:${integration.toLowerCase()}:${target}`;
}

async function _getCollectionGeneration(
  integration: string,
  target: string
): Promise<string> {
  if (_state.cache == null) return "0";
  const val = await _state.cache.get(_collectionGenKey(integration, target));
  return val != null ? String(val) : "0";
}

export async function vectorstoreCacheKey(
  integration: string,
  target: string,
  query: unknown,
  topK: unknown,
  filterVal: unknown
): Promise<string> {
  await _ensureInitialized();
  const gen = await _getCollectionGeneration(integration, target ?? "");
  return makeKey("vectorstore", integration, target ?? "", gen, query ?? "", topK, filterVal);
}

export async function invalidateVectorstoreCache(
  integration: string,
  target: string
): Promise<void> {
  await _ensureInitialized();
  if (_state.cache == null) return;
  await _state.cache.set(
    _collectionGenKey(integration, target ?? ""),
    String(Date.now() / 1000),
    0
  );
}

export async function lookupVectorstoreCache(
  cacheKey: string
): Promise<Record<string, unknown> | null> {
  await _ensureInitialized();
  if (_state.cache == null) return null;
  const raw = await _state.cache.get(cacheKey);
  if (raw != null && typeof raw === "object") return raw as Record<string, unknown>;
  return null;
}

export async function populateVectorstoreCache(
  cacheKey: string,
  result: Record<string, unknown>
): Promise<void> {
  if (_state.cache == null) return;
  await _state.cache.set(cacheKey, { type: "vectorstore", result });
}

// ---------------------------------------------------------------------------
// MCP list_tools() cache (keyed by server URL)
// ---------------------------------------------------------------------------

function _mcpListToolsKey(serverUrl: string): string {
  return makeKey("mcp_list_tools", serverUrl);
}

/** Return the cached list_tools() result for this MCP server, or null on miss. */
export async function lookupMcpToolsCache(serverUrl: string): Promise<unknown[] | null> {
  await _ensureInitialized();
  if (_state.cache == null) return null;
  const raw = await _state.cache.get(_mcpListToolsKey(serverUrl));
  if (raw != null && typeof raw === "object" && (raw as Record<string, unknown>)["type"] === "mcp_list_tools") {
    return ((raw as Record<string, unknown>)["tools"] as unknown[]) ?? null;
  }
  return null;
}

/** Cache the list_tools() response for this MCP server. */
export async function populateMcpToolsCache(serverUrl: string, tools: unknown[]): Promise<void> {
  if (_state.cache == null) return;
  await _state.cache.set(_mcpListToolsKey(serverUrl), { type: "mcp_list_tools", tools });
}

/** Evict the cached list_tools() for this server — called on re-connect (server restart). */
export async function invalidateMcpToolsCache(serverUrl: string): Promise<void> {
  await _ensureInitialized();
  if (_state.cache == null) return;
  await _state.cache.delete(_mcpListToolsKey(serverUrl));
}

// ---------------------------------------------------------------------------
// MCP call_tool() cache (keyed by server URL + tool name + args hash)
// ---------------------------------------------------------------------------

function _mcpCallKey(serverUrl: string, toolName: string, argsJson: string): string {
  return makeKey("mcp_call", serverUrl, toolName, argsJson);
}

/**
 * Return the cached call_tool() payload for (serverUrl, toolName, args), or null
 * on miss. Returns the full payload so the caller can reconstruct the result.
 */
export async function lookupMcpCallCache(
  serverUrl: string,
  toolName: string,
  args: Record<string, unknown> | string
): Promise<Record<string, unknown> | null> {
  await _ensureInitialized();
  if (_state.cache == null) return null;
  const raw = await _state.cache.get(_mcpCallKey(serverUrl, toolName, _normalizeArgs(args)));
  if (raw != null && typeof raw === "object" && (raw as Record<string, unknown>)["type"] === "mcp_call") {
    return raw as Record<string, unknown>;
  }
  return null;
}

/** Store an MCP call_tool() result in the cache. */
export async function populateMcpCallCache(
  serverUrl: string,
  toolName: string,
  args: Record<string, unknown> | string,
  content: unknown[],
  isError = false
): Promise<void> {
  if (_state.cache == null) return;
  await _state.cache.set(_mcpCallKey(serverUrl, toolName, _normalizeArgs(args)), {
    type: "mcp_call",
    content,
    isError,
  });
}
