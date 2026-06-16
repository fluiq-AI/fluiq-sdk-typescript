/**
 * OpenAI integration — patches chat.completions.create (sync/stream) and
 * optionally the responses API. Patches are applied lazily at instrument()
 * time and silently skipped if the `openai` package is not installed.
 */
import { randomUUID } from "crypto";
import { logTrace } from "../tracer";
import { _config } from "../config";
import { TraceType, toPlainObject } from "./shared/models";
import {
  currentParentId,
  currentLlmTraceId,
  withLlmTraceId,
  runInRootContext,
  isInLangchainLlm,
} from "./shared/context";
import { preCallGuard } from "./shared/securityGate";
import { preCallOptimize } from "./shared/optimizeGate";
import { learnFromOpenAIMessages } from "./shared/toolCache";
import { FluiqSecurityError } from "../exceptions";

/** Resolve a property off either a plain object or a class instance. */
function _get(obj: unknown, key: string): unknown {
  if (obj == null || typeof obj !== "object") return undefined;
  return (obj as Record<string, unknown>)[key];
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const MEDIA_TYPES = new Set([
  "image_url", "image", "input_image", "output_image",
  "input_audio", "audio", "output_audio", "video",
]);

function _toJsonable(obj: unknown): unknown {
  return _toJsonableInner(obj, new WeakSet<object>(), 0);
}

// Cycle- and depth-guarded so circular/exotic values can't overflow the stack
// while building a trace. The SDK must never crash the host application.
function _toJsonableInner(obj: unknown, seen: WeakSet<object>, depth: number): unknown {
  if (obj == null) return null;
  if (typeof obj !== "object") return obj;
  if (seen.has(obj as object)) return "[Circular]";
  if (depth > 8) return "[Truncated]";
  seen.add(obj as object);
  try {
    if (Array.isArray(obj)) return obj.map((v) => _toJsonableInner(v, seen, depth + 1));
    const o = obj as Record<string, unknown>;
    return Object.fromEntries(
      Object.entries(o).map(([k, v]) => [k, _toJsonableInner(v, seen, depth + 1)])
    );
  } finally {
    seen.delete(obj as object);
  }
}

function _stripMedia(content: unknown): unknown {
  if (content == null) return null;
  if (!Array.isArray(content)) return content;
  const kept: unknown[] = [];
  for (const part of content) {
    const ptype =
      typeof part === "object" && part !== null
        ? (part as Record<string, unknown>)["type"]
        : undefined;
    if (typeof ptype === "string" && MEDIA_TYPES.has(ptype)) continue;
    kept.push(_toJsonable(part));
  }
  return kept.length > 0 ? kept : null;
}

function _extractText(choices: unknown[]): string | null {
  for (const choice of choices) {
    const c = choice as Record<string, unknown>;
    const message = c["message"] as Record<string, unknown> | undefined;
    const content = message?.["content"];
    const stripped = _stripMedia(content);
    if (stripped != null) {
      if (typeof stripped === "string") return stripped;
      if (Array.isArray(stripped)) {
        return stripped
          .filter(
            (b): b is Record<string, unknown> =>
              typeof b === "object" && b !== null && (b as Record<string, unknown>)["type"] === "text"
          )
          .map((b) => String((b as Record<string, unknown>)["text"] ?? ""))
          .join("\n") || null;
      }
    }
  }
  return null;
}

function _extractToolCalls(choices: unknown[]): unknown[] | null {
  const calls: unknown[] = [];
  for (const choice of choices) {
    const c = choice as Record<string, unknown>;
    const message = c["message"] as Record<string, unknown> | undefined;
    const tcs = message?.["tool_calls"];
    if (Array.isArray(tcs)) calls.push(...tcs.map(_toJsonable));
  }
  return calls.length > 0 ? calls : null;
}

function _finishReasons(choices: unknown[]): unknown[] | null {
  const reasons = choices.map((c) => (c as Record<string, unknown>)["finish_reason"] ?? null);
  return reasons.length > 0 ? reasons : null;
}

function _extractUsage(response: Record<string, unknown>): Record<string, number> | null {
  const usage = response["usage"] as Record<string, unknown> | undefined;
  if (!usage) return null;
  return {
    prompt: Number(usage["prompt_tokens"] ?? 0),
    completion: Number(usage["completion_tokens"] ?? 0),
    total: Number(usage["total_tokens"] ?? 0),
  };
}

/** Cached prompt tokens reported under usage.prompt_tokens_details.cached_tokens. */
function _promptCachedTokens(usage: unknown): number | null {
  const details = _get(usage, "prompt_tokens_details");
  const cached = _get(details, "cached_tokens");
  return typeof cached === "number" ? cached : null;
}

/** Per-choice reasoning ("thinking") + reasoning-token count, mirroring thinking_trace.py. */
function _extractThinking(choices: unknown[], usage: unknown): unknown[] | null {
  const items: unknown[] = [];
  for (const choice of choices ?? []) {
    const message = _get(choice, "message");
    const reasoning = _get(message, "reasoning_content") ?? _get(message, "reasoning");
    if (reasoning) items.push({ reasoning });
  }
  const details = _get(usage, "completion_tokens_details");
  const reasoningTokens = _get(details, "reasoning_tokens");
  if (reasoningTokens) items.push({ reasoning_tokens: reasoningTokens });
  return items.length > 0 ? items : null;
}

const MCP_TOOL_TYPE = "mcp";
const MCP_OUTPUT_TYPES = new Set(["mcp_list_tools", "mcp_call", "mcp_approval_request"]);

function _extractMcpServersFromTools(tools: unknown): unknown[] | null {
  if (!Array.isArray(tools)) return null;
  const out = tools.filter((t) => _get(t, "type") === MCP_TOOL_TYPE).map(_toJsonable);
  return out.length > 0 ? out : null;
}

function _extractMcpCallsFromOutput(output: unknown): unknown[] | null {
  if (!Array.isArray(output)) return null;
  const out = output
    .filter((item) => MCP_OUTPUT_TYPES.has(String(_get(item, "type"))))
    .map(_toJsonable);
  return out.length > 0 ? out : null;
}

/** JSON-safe snapshot for endpoint traces (embeddings/images/audio). */
function _safeJsonable(obj: unknown): unknown {
  if (obj == null) return null;
  try {
    return JSON.parse(JSON.stringify(_toJsonable(obj)));
  } catch {
    return { type: typeof obj };
  }
}

// ---------------------------------------------------------------------------
// Pending tool call latency tracking
// ---------------------------------------------------------------------------

const _pendingToolCalls = new Map<string, { ts: number; name: string | null }>();

function _recordDispatchedToolCalls(choices: unknown[]): void {
  const now = Date.now() / 1000;
  for (const choice of choices) {
    const c = choice as Record<string, unknown>;
    const message = c["message"] as Record<string, unknown> | undefined;
    const tcs = message?.["tool_calls"];
    if (!Array.isArray(tcs)) continue;
    for (const tc of tcs) {
      const t = tc as Record<string, unknown>;
      const id = String(t["id"] ?? "");
      const fn = t["function"] as Record<string, unknown> | undefined;
      const name = typeof fn?.["name"] === "string" ? fn["name"] : null;
      if (id) _pendingToolCalls.set(id, { ts: now, name });
    }
  }
}

function _gcPendingToolCalls(ttl = 3600): void {
  const cutoff = Date.now() / 1000 - ttl;
  for (const [id, { ts }] of _pendingToolCalls) {
    if (ts < cutoff) _pendingToolCalls.delete(id);
  }
}

function _computeToolCallLatencies(
  messages: unknown[] | null | undefined
): unknown[] | null {
  if (!messages) return null;
  const now = Date.now() / 1000;
  const latencies: unknown[] = [];
  for (const msg of messages) {
    if (typeof msg !== "object" || msg === null) continue;
    const m = msg as Record<string, unknown>;
    if (m["role"] !== "tool") continue;
    const tcId = String(m["tool_call_id"] ?? "");
    const entry = _pendingToolCalls.get(tcId);
    if (!entry) continue;
    _pendingToolCalls.delete(tcId);
    latencies.push({ tool_call_id: tcId, name: entry.name, latency: now - entry.ts });
  }
  return latencies.length > 0 ? latencies : null;
}

// ---------------------------------------------------------------------------
// Emit helpers
// ---------------------------------------------------------------------------

async function _emitChatTrace(
  params: Record<string, unknown>,
  response: Record<string, unknown>,
  start: number,
  end: number,
  toolCallLatencies: unknown
): Promise<void> {
  const usage = response["usage"];
  const choices = (response["choices"] as unknown[]) ?? [];
  _recordDispatchedToolCalls(choices);
  const text = _extractText(choices);
  const toolCalls = _extractToolCalls(choices);
  const tokens = _extractUsage(response);

  await logTrace(
    toPlainObject({
      type: "llm",
      integration: TraceType.OpenAI,
      model: (params["model"] as string) ?? (response["model"] as string) ?? null,
      messages: _toJsonable(params["messages"]) as unknown[],
      tools: _toJsonable(params["tools"]) as unknown,
      tool_choice: _toJsonable(params["tool_choice"]) as unknown,
      response: text,
      thinking: _extractThinking(choices, usage),
      tool_calls: toolCalls,
      tool_call_latencies: toolCallLatencies,
      finish_reasons: _finishReasons(choices),
      latency: end - start,
      parent_id: currentParentId(),
      tokens: tokens
        ? { prompt: tokens.prompt, completion: tokens.completion, total: tokens.total }
        : null,
      prompt_cached_tokens: _promptCachedTokens(usage),
    })
  );
}

// ---------------------------------------------------------------------------
// Responses API emit helpers
// ---------------------------------------------------------------------------

async function _emitResponsesTrace(
  params: Record<string, unknown>,
  response: Record<string, unknown>,
  start: number,
  end: number
): Promise<void> {
  const output = response["output"];
  await logTrace(
    toPlainObject({
      type: "llm",
      integration: TraceType.OpenAI,
      api: "responses",
      model: (params["model"] as string) ?? (response["model"] as string) ?? null,
      input: _toJsonable(params["input"]),
      tools: _toJsonable(params["tools"]),
      mcp_servers: _extractMcpServersFromTools(params["tools"]),
      mcp_calls: _extractMcpCallsFromOutput(output),
      response: _toJsonable(output),
      latency: end - start,
      parent_id: currentParentId(),
      tokens: _toJsonable(response["usage"]) as unknown as import("./shared/models").Tokens | null,
    })
  );
}

async function _emitResponsesError(
  params: Record<string, unknown>,
  error: Error,
  start: number,
  end: number,
  api = "responses"
): Promise<void> {
  await logTrace(
    toPlainObject({
      type: "llm",
      integration: TraceType.OpenAI,
      api,
      model: params["model"] as string,
      input: _toJsonable(params["input"]),
      tools: _toJsonable(params["tools"]),
      output: String(error),
      latency: end - start,
      parent_id: currentParentId(),
      success: false,
    })
  );
}

async function _emitChatError(
  params: Record<string, unknown>,
  error: Error,
  start: number,
  end: number,
  api = "chat.completions"
): Promise<void> {
  await logTrace(
    toPlainObject({
      type: "llm",
      integration: TraceType.OpenAI,
      api,
      model: params["model"] as string,
      messages: _toJsonable(params["messages"]) as unknown[],
      tools: _toJsonable(params["tools"]) as unknown,
      tool_choice: _toJsonable(params["tool_choice"]) as unknown,
      output: String(error),
      latency: end - start,
      parent_id: currentParentId(),
      success: false,
    })
  );
}

async function _emitSecurityBlockedTrace(
  params: Record<string, unknown>,
  exc: FluiqSecurityError,
  start: number,
  end: number,
  api = "chat.completions"
): Promise<void> {
  const attackTypes = exc.attackTypes;
  const has = (t: string) => attackTypes.includes(t);

  const base: Record<string, unknown> =
    api === "responses"
      ? {
          type: "llm",
          integration: TraceType.OpenAI,
          api,
          model: params["model"] as string,
          input: _toJsonable(params["input"]),
          tools: _toJsonable(params["tools"]),
          latency: end - start,
          parent_id: currentParentId(),
          success: false,
        }
      : {
          type: "llm",
          integration: TraceType.OpenAI,
          api,
          model: params["model"] as string,
          messages: _toJsonable(params["messages"]),
          tools: _toJsonable(params["tools"]),
          tool_choice: _toJsonable(params["tool_choice"]),
          latency: end - start,
          parent_id: currentParentId(),
          success: false,
        };

  await logTrace({
    ...base,
    _security_pre_blocked: true,
    status: "blocked",
    security_risk_level: exc.riskLevel,
    security_risk_score: 1.0,
    should_block: true,
    block_reason: exc.blockReason,
    injection_detected: has("prompt_injection"),
    injection_patterns: has("prompt_injection") ? ["prompt_injection"] : [],
    jailbreak_detected: has("jailbreak"),
    jailbreak_patterns: has("jailbreak") ? ["jailbreak"] : [],
    skeleton_key_detected: has("skeleton_key"),
    skeleton_key_patterns: has("skeleton_key") ? ["skeleton_key"] : [],
    secrets_detected: false,
    secret_types: [],
    pii_entities_prompt: [],
    pii_entities_response: [],
    prompt_redacted: "",
    response_redacted: "",
    indirect_injection_detected: false,
    indirect_injection_sources: [],
    semantic_attack_score: 0.0,
  });
}

// ---------------------------------------------------------------------------
// Patch: chat.completions.create
// ---------------------------------------------------------------------------

export function patchOpenAI(): void {
  let Completions: { prototype: Record<string, unknown> } | null = null;
  try {
    Completions = require("openai/resources/chat/completions")
      .Completions as { prototype: Record<string, unknown> };
  } catch {
    return; // openai not installed
  }

  const original = Completions.prototype["create"] as (
    this: unknown,
    params: unknown,
    opts?: unknown
  ) => Promise<unknown>;

  if ((original as unknown as Record<string, boolean>)["_fluiq_patched"]) return;

  // NOTE: intentionally NOT `async`. The OpenAI SDK's `create()` returns an
  // `APIPromise` (with `_thenUnwrap`, `.withResponse()`, …) and callers such as
  // `beta.chat.completions.parse()` call `create(...)._thenUnwrap(...)` on it.
  // An `async function` would re-wrap that APIPromise in a plain Promise and
  // strip those methods. Keeping `wrapped` synchronous lets the pass-through
  // paths return the original APIPromise untouched; the traced path still
  // returns a Promise via `runInRootContext(async …)`.
  function wrapped(
    this: unknown,
    params: Record<string, unknown>,
    opts?: unknown
  ): Promise<unknown> | unknown {
    if (isInLangchainLlm()) return original.call(this, params, opts);
    // Already inside another Fluiq LLM trace (e.g. chat.completions.parse() or a
    // stream helper that delegates to create()) — that outer wrapper traces this
    // call, so pass through to avoid double-tracing.
    if (currentLlmTraceId()) return original.call(this, params, opts);
    return runInRootContext(async () => {
      _gcPendingToolCalls();
      const toolCallLatencies = _computeToolCallLatencies(
        params["messages"] as unknown[] | undefined
      );
      const traceId = randomUUID();
      const start = Date.now() / 1000;

      // Emit running trace
      await logTrace(
        toPlainObject({
          type: "llm",
          integration: TraceType.OpenAI,
          api: "chat.completions",
          trace_id: traceId,
          model: params["model"] as string,
          messages: _toJsonable(params["messages"]) as unknown[],
          status: "running",
          started_at: start,
          parent_id: currentParentId(),
        })
      );

      return withLlmTraceId(traceId, async () => {
        try {
          await preCallGuard(params);
        } catch (secExc) {
          if (secExc instanceof FluiqSecurityError) {
            await _emitSecurityBlockedTrace(params, secExc, start, Date.now() / 1000);
          }
          throw secExc;
        }

        try {
          learnFromOpenAIMessages(params["messages"]);
        } catch {
          // tool-cache learning must never break the call
        }

        const cached = await preCallOptimize(params, "openai");
        if (cached != null) {
          const end = Date.now() / 1000;
          await logTrace({
            type: "llm",
            integration: TraceType.OpenAI,
            api: "chat.completions",
            trace_id: traceId,
            model: params["model"],
            messages: _toJsonable(params["messages"]),
            tools: _toJsonable(params["tools"]),
            response: cached["response"],
            tool_calls: cached["tool_calls"],
            mcp_calls: cached["mcp_calls"],
            latency: end - start,
            parent_id: currentParentId(),
            _cache_hit: true,
            tokens: null,
          });
          return _buildOpenAICachedResponse(cached, params);
        }

        let response: unknown;
        try {
          response = await original.call(this, params, opts);
        } catch (e) {
          await _emitChatError(params, e as Error, start, Date.now() / 1000);
          throw e;
        }

        // Handle streaming responses — return as-is and let the caller consume
        if (params["stream"]) {
          return _wrapStream(response, params, start, traceId, toolCallLatencies);
        }

        const end = Date.now() / 1000;
        await _emitChatTrace(
          params,
          response as Record<string, unknown>,
          start,
          end,
          toolCallLatencies
        );
        return response;
      });
    });
  }

  (wrapped as unknown as Record<string, boolean>)["_fluiq_patched"] = true;
  Completions.prototype["create"] = wrapped;
}

// ---------------------------------------------------------------------------
// Streaming wrapper (async generator pass-through with accumulation)
// ---------------------------------------------------------------------------

function _wrapStream(
  stream: unknown,
  params: Record<string, unknown>,
  start: number,
  traceId: string,
  toolCallLatencies: unknown
): AsyncIterable<unknown> {
  const chunks: unknown[] = [];
  const originalStream = stream as AsyncIterable<unknown>;
  // When secure mode='block', buffer the whole stream before yielding any
  // chunk so the response gate can fire (and raise) before the caller sees
  // content — mirroring _wrap_chat_stream in the Python SDK.
  const needsGate = _config.secure && _config.secure_mode === "block";

  async function emitEnd(): Promise<void> {
    const end = Date.now() / 1000;
    const accumulated = _accumulateChunks(chunks);
    await withLlmTraceId(traceId, () =>
      logTrace(
        toPlainObject({
          type: "llm",
          integration: TraceType.OpenAI,
          api: "chat.completions.stream",
          trace_id: traceId,
          model: (params["model"] as string) ?? accumulated.model,
          messages: _toJsonable(params["messages"]) as unknown[],
          tools: _toJsonable(params["tools"]) as unknown,
          response: accumulated.text,
          thinking: accumulated.thinking,
          tool_calls: accumulated.toolCalls,
          tool_call_latencies: toolCallLatencies,
          finish_reasons: accumulated.finishReasons,
          latency: end - start,
          parent_id: currentParentId(),
          tokens: accumulated.usage as unknown as import("./shared/models").Tokens | null,
          prompt_cached_tokens: accumulated.promptCachedTokens,
        })
      )
    );
  }

  async function* bufferedGenerator(): AsyncGenerator<unknown> {
    try {
      for await (const chunk of originalStream) chunks.push(chunk);
    } catch (e) {
      await _emitChatError(params, e as Error, start, Date.now() / 1000, "chat.completions.stream");
      throw e;
    }
    // May throw FluiqSecurityError via the response gate before any yield.
    await emitEnd();
    for (const chunk of chunks) yield chunk;
  }

  async function* generator(): AsyncGenerator<unknown> {
    try {
      for await (const chunk of originalStream) {
        chunks.push(chunk);
        yield chunk;
      }
    } catch (e) {
      await _emitChatError(params, e as Error, start, Date.now() / 1000, "chat.completions.stream");
      throw e;
    }
    // Stream ended — emit accumulated trace
    await emitEnd();
  }

  // Preserve the original Stream object (notably `.controller`, read by
  // ChatCompletionStream._createChatCompletion) and only override iteration.
  // A Proxy keeps method `this`-binding intact so the SDK's ES private fields
  // keep working — unlike Object.create / a bare generator.
  const makeIterator = () => (needsGate ? bufferedGenerator() : generator());
  return new Proxy(stream as object, {
    get(target, prop) {
      if (prop === Symbol.asyncIterator) return makeIterator;
      const value = Reflect.get(target, prop, target);
      return typeof value === "function"
        ? (value as (...a: unknown[]) => unknown).bind(target)
        : value;
    },
  }) as AsyncIterable<unknown>;
}

interface Accumulated {
  text: string | null;
  model: string | null;
  toolCalls: unknown[] | null;
  finishReasons: unknown[] | null;
  usage: Record<string, number> | null;
  thinking: unknown[] | null;
  promptCachedTokens: number | null;
}

function _accumulateChunks(chunks: unknown[]): Accumulated {
  let text = "";
  let model: string | null = null;
  const toolCallAccum: Record<number, Record<string, unknown>> = {};
  const finishReasons: unknown[] = [];
  const thinkingParts: string[] = [];
  let usage: Record<string, number> | null = null;
  let promptCachedTokens: number | null = null;

  for (const chunk of chunks) {
    const c = chunk as Record<string, unknown>;
    if (c["model"] && typeof c["model"] === "string") model = c["model"];

    const u = c["usage"] as Record<string, unknown> | undefined;
    if (u) {
      usage = {
        prompt: Number(u["prompt_tokens"] ?? 0),
        completion: Number(u["completion_tokens"] ?? 0),
        total: Number(u["total_tokens"] ?? 0),
      };
      const cached = _promptCachedTokens(u);
      if (cached != null) promptCachedTokens = cached;
    }

    const choices = (c["choices"] as unknown[]) ?? [];
    for (const choice of choices) {
      const ch = choice as Record<string, unknown>;
      const delta = (ch["delta"] as Record<string, unknown>) ?? {};

      if (typeof delta["content"] === "string") text += delta["content"];

      const reasoning = delta["reasoning"] ?? delta["reasoning_content"];
      if (typeof reasoning === "string" && reasoning) thinkingParts.push(reasoning);

      const tcs = delta["tool_calls"] as Array<Record<string, unknown>> | undefined;
      if (tcs) {
        for (const tc of tcs) {
          const idx = Number(tc["index"] ?? 0);
          if (!toolCallAccum[idx]) {
            toolCallAccum[idx] = { id: tc["id"], type: tc["type"], function: { name: "", arguments: "" } };
          }
          const acc = toolCallAccum[idx];
          const fn = tc["function"] as Record<string, unknown> | undefined;
          const accFn = acc["function"] as Record<string, unknown>;
          if (fn?.["name"]) accFn["name"] = String(accFn["name"] ?? "") + String(fn["name"]);
          if (fn?.["arguments"]) accFn["arguments"] = String(accFn["arguments"] ?? "") + String(fn["arguments"]);
        }
      }

      if (ch["finish_reason"]) finishReasons.push(ch["finish_reason"]);
    }
  }

  const toolCalls = Object.values(toolCallAccum);
  return {
    text: text || null,
    model,
    toolCalls: toolCalls.length > 0 ? toolCalls : null,
    finishReasons: finishReasons.length > 0 ? finishReasons : null,
    usage,
    thinking: thinkingParts.length > 0 ? thinkingParts : null,
    promptCachedTokens,
  };
}

function _buildOpenAICachedResponse(
  payload: Record<string, unknown>,
  params: Record<string, unknown>
): Record<string, unknown> {
  const text = payload["response"] as string | null;
  const toolCalls = payload["tool_calls"] as unknown[] | null;
  const tcObjs = toolCalls?.map((tc) => {
    const t = tc as Record<string, unknown>;
    const fn = (t["function"] as Record<string, unknown>) ?? {};
    return {
      id: t["id"],
      type: t["type"] ?? "function",
      function: { name: fn["name"], arguments: fn["arguments"] ?? "" },
    };
  }) ?? null;

  return {
    choices: [
      {
        message: { content: text ?? null, role: "assistant", tool_calls: tcObjs ?? null, refusal: null },
        finish_reason: tcObjs ? "tool_calls" : "stop",
        index: 0,
        logprobs: null,
      },
    ],
    model: params["model"] ?? "",
    id: "fluiq-cached",
    object: "chat.completion",
    // Served from cache — no provider call was made, so all token counts are 0.
    // Use a zeroed object (not null) so caller code that reads e.g.
    // `usage.prompt_tokens` or `usage.prompt_tokens_details.cached_tokens`
    // doesn't throw on a cache hit.
    usage: {
      prompt_tokens: 0,
      completion_tokens: 0,
      total_tokens: 0,
      prompt_tokens_details: { cached_tokens: 0 },
    },
    _fluiq_cached: true,
  };
}

// ---------------------------------------------------------------------------
// Generic prototype patcher
// ---------------------------------------------------------------------------

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyFn = (this: unknown, ...args: any[]) => unknown;

/**
 * Patch `ClassName.prototype[method]` from the first resolvable module path.
 * No-ops when the openai package (or the specific method) is not present, and
 * is idempotent via a `_fluiq_patched` marker. Mirrors the defensive
 * `if not hasattr(...)` guards in the Python SDK.
 */
function _patchProto(
  modPaths: string[],
  className: string,
  method: string,
  wrap: (orig: AnyFn) => AnyFn
): void {
  for (const p of modPaths) {
    let mod: Record<string, unknown>;
    try {
      mod = require(p) as Record<string, unknown>;
    } catch {
      continue; // try next candidate path
    }
    const cls = mod[className] as { prototype?: Record<string, unknown> } | undefined;
    if (!cls || !cls.prototype) continue;
    const orig = cls.prototype[method];
    if (typeof orig !== "function") return; // method not supported in this version
    if ((orig as unknown as Record<string, boolean>)["_fluiq_patched"]) return;
    const wrapped = wrap(orig as AnyFn);
    (wrapped as unknown as Record<string, boolean>)["_fluiq_patched"] = true;
    cls.prototype[method] = wrapped;
    return;
  }
}

// ---------------------------------------------------------------------------
// Patch: responses.create (+ streaming)
// ---------------------------------------------------------------------------

export function patchOpenAIResponses(): void {
  _patchProto(
    ["openai/resources/responses/responses", "openai/resources/responses"],
    "Responses",
    "create",
    (original) =>
      async function (this: unknown, params: Record<string, unknown>, opts?: unknown) {
        if (isInLangchainLlm()) return original.call(this, params, opts);
        return runInRootContext(async () => {
          const traceId = randomUUID();
          const start = Date.now() / 1000;
          await logTrace(
            toPlainObject({
              type: "llm",
              integration: TraceType.OpenAI,
              api: "responses",
              trace_id: traceId,
              model: params["model"] as string,
              input: _toJsonable(params["input"]),
              status: "running",
              started_at: start,
              parent_id: currentParentId(),
            })
          );
          return withLlmTraceId(traceId, async () => {
            try {
              await preCallGuard(params);
            } catch (secExc) {
              if (secExc instanceof FluiqSecurityError) {
                await _emitSecurityBlockedTrace(params, secExc, start, Date.now() / 1000, "responses");
              }
              throw secExc;
            }
            let response: unknown;
            try {
              response = await original.call(this, params, opts);
            } catch (e) {
              await _emitResponsesError(params, e as Error, start, Date.now() / 1000);
              throw e;
            }
            if (params["stream"]) {
              return _wrapResponsesStream(response, params, start, traceId);
            }
            await _emitResponsesTrace(params, response as Record<string, unknown>, start, Date.now() / 1000);
            return response;
          });
        });
      }
  );
}

function _wrapResponsesStream(
  stream: unknown,
  params: Record<string, unknown>,
  start: number,
  traceId: string
): AsyncIterable<unknown> {
  const original = stream as AsyncIterable<unknown>;
  let finalResponse: Record<string, unknown> | null = null;

  async function* generator(): AsyncGenerator<unknown> {
    try {
      for await (const event of original) {
        if (_get(event, "type") === "response.completed") {
          finalResponse = (_get(event, "response") as Record<string, unknown>) ?? finalResponse;
        }
        yield event;
      }
      const end = Date.now() / 1000;
      const output = finalResponse ? finalResponse["output"] : null;
      await withLlmTraceId(traceId, () =>
        logTrace(
          toPlainObject({
            type: "llm",
            integration: TraceType.OpenAI,
            api: "responses.stream",
            trace_id: traceId,
            model: (params["model"] as string) ?? (finalResponse?.["model"] as string) ?? null,
            input: _toJsonable(params["input"]),
            tools: _toJsonable(params["tools"]),
            mcp_servers: _extractMcpServersFromTools(params["tools"]),
            mcp_calls: _extractMcpCallsFromOutput(output),
            response: _toJsonable(output),
            latency: end - start,
            parent_id: currentParentId(),
            tokens: _toJsonable(finalResponse?.["usage"]) as unknown as import("./shared/models").Tokens | null,
          })
        )
      );
    } catch (e) {
      await _emitResponsesError(params, e as Error, start, Date.now() / 1000, "responses.stream");
      throw e;
    }
  }

  return generator();
}

// ---------------------------------------------------------------------------
// Patch: chat.completions.parse (structured outputs)
// ---------------------------------------------------------------------------

export function patchOpenAIParse(): void {
  const wrap =
    (original: AnyFn) =>
      async function (this: unknown, params: Record<string, unknown>, opts?: unknown) {
        if (isInLangchainLlm()) return original.call(this, params, opts);
        return runInRootContext(async () => {
          _gcPendingToolCalls();
          const toolCallLatencies = _computeToolCallLatencies(params["messages"] as unknown[] | undefined);
          const traceId = randomUUID();
          const start = Date.now() / 1000;
          await logTrace(
            toPlainObject({
              type: "llm",
              integration: TraceType.OpenAI,
              api: "chat.completions.parse",
              trace_id: traceId,
              model: params["model"] as string,
              messages: _toJsonable(params["messages"]) as unknown[],
              status: "running",
              started_at: start,
              parent_id: currentParentId(),
            })
          );
          return withLlmTraceId(traceId, async () => {
            try {
              await preCallGuard(params);
            } catch (secExc) {
              if (secExc instanceof FluiqSecurityError) {
                await _emitSecurityBlockedTrace(params, secExc, start, Date.now() / 1000, "chat.completions.parse");
              }
              throw secExc;
            }
            let response: unknown;
            try {
              response = await original.call(this, params, opts);
            } catch (e) {
              await _emitChatError(params, e as Error, start, Date.now() / 1000, "chat.completions.parse");
              throw e;
            }
            await _emitChatTrace(params, response as Record<string, unknown>, start, Date.now() / 1000, toolCallLatencies);
            return response;
          });
        });
      };
  // `parse` lives on the chat Completions resource (and historically under beta).
  _patchProto(["openai/resources/chat/completions"], "Completions", "parse", wrap);
  _patchProto(["openai/resources/beta/chat/completions"], "Completions", "parse", wrap);
}

// ---------------------------------------------------------------------------
// chat.completions.stream() helper
// ---------------------------------------------------------------------------
// Intentionally NOT patched. `chat.completions.stream()` returns a
// `ChatCompletionStream` whose `_createChatCompletion` calls
// `chat.completions.create({ stream: true })` internally — which is already
// patched above. Patching the helper too would double-trace, and wrapping the
// returned ChatCompletionStream breaks its ES private fields (`.on()` /
// `.finalChatCompletion()`). Letting the inner `create` handle it yields a
// single trace and keeps the ChatCompletionStream API fully intact.
export function patchOpenAIStreamHelper(): void {
  /* no-op — see note above */
}

// ---------------------------------------------------------------------------
// Endpoints: embeddings / images / audio
// ---------------------------------------------------------------------------

async function _emitEndpointTrace(
  api: string,
  params: Record<string, unknown>,
  response: unknown,
  start: number,
  end: number,
  redactInput?: unknown,
  redactResponse?: (r: unknown) => unknown
): Promise<void> {
  await logTrace(
    toPlainObject({
      type: "llm",
      integration: TraceType.OpenAI,
      api,
      model: (params["model"] as string) ?? (_get(response, "model") as string) ?? null,
      input:
        redactInput !== undefined
          ? redactInput
          : _safeJsonable(params["input"] ?? params["prompt"]),
      response: redactResponse ? redactResponse(response) : _safeJsonable(response),
      latency: end - start,
      parent_id: currentParentId(),
      tokens: _safeJsonable(_get(response, "usage")) as unknown as import("./shared/models").Tokens | null,
    })
  );
}

/**
 * Summarize an image-generation/edit response for tracing. Image responses can
 * carry the full image inline as a ~1-2 MB `b64_json` base64 string; shipping
 * that in a trace bloats the payload (and can make the ingest endpoint fail).
 * Keep the useful metadata (url, revised_prompt, usage) and replace the raw
 * base64 with a short placeholder.
 */
function _summarizeImageResponse(response: unknown): unknown {
  const data = _get(response, "data");
  const items = Array.isArray(data)
    ? data.map((d) => {
        const o = (d ?? {}) as Record<string, unknown>;
        const out: Record<string, unknown> = {};
        if (o["url"] != null) out["url"] = o["url"];
        if (o["revised_prompt"] != null) out["revised_prompt"] = o["revised_prompt"];
        if (typeof o["b64_json"] === "string") {
          out["b64_json"] = `<base64 image omitted: ${o["b64_json"].length} chars>`;
        }
        return out;
      })
    : data;
  return _safeJsonable({
    created: _get(response, "created"),
    size: _get(response, "size"),
    quality: _get(response, "quality"),
    output_format: _get(response, "output_format"),
    data: items,
    usage: _get(response, "usage"),
  });
}

function _summarizeAudioInput(params: Record<string, unknown>): unknown {
  const file = params["file"];
  if (file == null) return null;
  const name = _get(file, "name");
  return name ? { filename: name } : { type: typeof file };
}

function _makeEndpointWrapper(
  api: string,
  redact?: (p: Record<string, unknown>) => unknown,
  redactResponse?: (r: unknown) => unknown
) {
  return (original: AnyFn) =>
    async function (this: unknown, params: Record<string, unknown>, opts?: unknown) {
      if (isInLangchainLlm()) return original.call(this, params, opts);
      const start = Date.now() / 1000;
      const response = await original.call(this, params, opts);
      const end = Date.now() / 1000;
      await _emitEndpointTrace(
        api,
        params,
        response,
        start,
        end,
        redact ? redact(params) : undefined,
        redactResponse
      );
      return response;
    };
}

export function patchOpenAIEmbeddings(): void {
  _patchProto(["openai/resources/embeddings"], "Embeddings", "create", _makeEndpointWrapper("embeddings"));
}

export function patchOpenAIImages(): void {
  _patchProto(["openai/resources/images"], "Images", "generate", _makeEndpointWrapper("images.generate", undefined, _summarizeImageResponse));
  _patchProto(["openai/resources/images"], "Images", "edit", _makeEndpointWrapper("images.edit", undefined, _summarizeImageResponse));
  _patchProto(["openai/resources/images"], "Images", "createVariation", _makeEndpointWrapper("images.variation", undefined, _summarizeImageResponse));
}

export function patchOpenAIAudio(): void {
  _patchProto(
    ["openai/resources/audio/transcriptions"],
    "Transcriptions",
    "create",
    _makeEndpointWrapper("audio.transcriptions", _summarizeAudioInput)
  );
  _patchProto(
    ["openai/resources/audio/translations"],
    "Translations",
    "create",
    _makeEndpointWrapper("audio.translations", _summarizeAudioInput)
  );
  _patchProto(["openai/resources/audio/speech"], "Speech", "create", _makeEndpointWrapper("audio.speech"));
}
