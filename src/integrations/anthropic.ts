/**
 * Anthropic integration — patches messages.create (sync/stream).
 * Patches are applied lazily at instrument() time and silently skipped
 * if the `@anthropic-ai/sdk` package is not installed.
 */
import { randomUUID } from "crypto";
import { logTrace } from "../tracer";
import { _config } from "../config";
import { TraceType, toPlainObject } from "./shared/models";
import {
  currentParentId,
  withLlmTraceId,
  runInRootContext,
  isInLangchainLlm,
} from "./shared/context";
import { preCallGuard } from "./shared/securityGate";
import { preCallOptimize } from "./shared/optimizeGate";
import { learnFromAnthropicMessages } from "./shared/toolCache";
import { FluiqSecurityError } from "../exceptions";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

function _blockType(b: unknown): string | undefined {
  if (b == null || typeof b !== "object") return undefined;
  const t = (b as Record<string, unknown>)["type"];
  return typeof t === "string" ? t : undefined;
}

// MCP extraction --------------------------------------------------------------

const MCP_BLOCK_TYPES = new Set(["mcp_tool_use", "mcp_tool_result"]);

function _extractMcpServers(params: Record<string, unknown>): unknown[] | null {
  const servers = params["mcp_servers"];
  return servers ? (_toJsonable(servers) as unknown[]) : null;
}

function _extractMcpBlocks(content: unknown): unknown[] | null {
  if (!Array.isArray(content)) return null;
  const out = content.filter((b) => MCP_BLOCK_TYPES.has(String(_blockType(b)))).map(_toJsonable);
  return out.length > 0 ? out : null;
}

function _extractMcpResultsFromMessages(messages: unknown): unknown[] | null {
  if (!Array.isArray(messages)) return null;
  const results: unknown[] = [];
  for (const msg of messages) {
    if (typeof msg !== "object" || msg === null) continue;
    const m = msg as Record<string, unknown>;
    if (m["role"] !== "user" || !Array.isArray(m["content"])) continue;
    for (const block of m["content"]) {
      if (_blockType(block) === "mcp_tool_result") results.push(_toJsonable(block));
    }
  }
  return results.length > 0 ? results : null;
}

// Prompt-cache token extraction ----------------------------------------------

function _cacheReadTokens(usage: unknown): number | null {
  const v = usage && typeof usage === "object"
    ? (usage as Record<string, unknown>)["cache_read_input_tokens"]
    : undefined;
  return typeof v === "number" ? v : null;
}

function _cacheCreationTokens(usage: unknown): number | null {
  const v = usage && typeof usage === "object"
    ? (usage as Record<string, unknown>)["cache_creation_input_tokens"]
    : undefined;
  return typeof v === "number" ? v : null;
}

// Prompt cache_control injection (when fluiq.optimize() is active) -------------

function maybeInjectAnthropicCacheControl(params: Record<string, unknown>): void {
  if (!_config.optimize) return;
  _injectSystem(params);
  _injectLastTool(params);
}

function _injectSystem(params: Record<string, unknown>): void {
  const system = params["system"];
  if (!system) return;
  if (typeof system === "string") {
    params["system"] = [
      { type: "text", text: system, cache_control: { type: "ephemeral" } },
    ];
    return;
  }
  if (Array.isArray(system)) {
    const newSystem = [...system];
    for (let i = newSystem.length - 1; i >= 0; i--) {
      const block = newSystem[i];
      if (block && typeof block === "object" && !Array.isArray(block)) {
        const b = block as Record<string, unknown>;
        if (b["type"] === "text" && !b["cache_control"]) {
          newSystem[i] = { ...b, cache_control: { type: "ephemeral" } };
          break;
        }
      }
    }
    params["system"] = newSystem;
  }
}

function _injectLastTool(params: Record<string, unknown>): void {
  const tools = params["tools"];
  if (!Array.isArray(tools) || tools.length === 0) return;
  const last = tools[tools.length - 1];
  if (!last || typeof last !== "object" || Array.isArray(last)) return;
  const b = last as Record<string, unknown>;
  if (b["cache_control"]) return;
  params["tools"] = [...tools.slice(0, -1), { ...b, cache_control: { type: "ephemeral" } }];
}

const MEDIA_TYPES = new Set(["image", "document", "tool_result", "tool_use"]);

function _extractTextFromContent(content: unknown): string | null {
  if (!Array.isArray(content)) return null;
  const texts = content
    .filter((b): b is Record<string, unknown> => {
      if (typeof b !== "object" || b === null) return false;
      const block = b as Record<string, unknown>;
      return block["type"] === "text";
    })
    .map((b) => String((b as Record<string, unknown>)["text"] ?? ""));
  return texts.join("\n") || null;
}

function _extractToolUses(content: unknown): unknown[] | null {
  if (!Array.isArray(content)) return null;
  const uses = content.filter((b) => {
    if (typeof b !== "object" || b === null) return false;
    return (b as Record<string, unknown>)["type"] === "tool_use";
  });
  return uses.length > 0 ? uses.map(_toJsonable) : null;
}

const THINKING_BLOCK_TYPES = new Set(["thinking", "redacted_thinking"]);

function _extractThinking(content: unknown): unknown[] | null {
  if (!Array.isArray(content)) return null;
  const blocks: unknown[] = [];
  for (const block of content) {
    if (!THINKING_BLOCK_TYPES.has(String(_blockType(block)))) continue;
    const dumped = _toJsonable(block);
    if (dumped && typeof dumped === "object" && !Array.isArray(dumped)) {
      const d = dumped as Record<string, unknown>;
      blocks.push({
        type: d["type"],
        thinking: d["thinking"],
        signature: d["signature"],
        data: d["data"],
      });
    } else {
      blocks.push(dumped);
    }
  }
  return blocks.length > 0 ? blocks : null;
}

function _extractUsage(response: Record<string, unknown>): Record<string, number | null> | null {
  const usage = response["usage"] as Record<string, unknown> | undefined;
  if (!usage) return null;
  const inp = Number(usage["input_tokens"] ?? 0);
  const out = Number(usage["output_tokens"] ?? 0);
  return { prompt: inp, completion: out, total: inp + out };
}

// ---------------------------------------------------------------------------
// Pending tool call latency tracking (for Anthropic tool_result messages)
// ---------------------------------------------------------------------------

const _pendingToolCalls = new Map<string, { ts: number; name: string | null }>();

function _recordDispatchedToolCalls(content: unknown): void {
  if (!Array.isArray(content)) return;
  const now = Date.now() / 1000;
  for (const block of content) {
    if (typeof block !== "object" || block === null) continue;
    const b = block as Record<string, unknown>;
    if (b["type"] === "tool_use" && typeof b["id"] === "string") {
      _pendingToolCalls.set(b["id"], { ts: now, name: String(b["name"] ?? "") || null });
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
    // Anthropic tool results are in role="user" messages with type="tool_result"
    if (m["role"] !== "user") continue;
    const content = m["content"];
    if (!Array.isArray(content)) continue;
    for (const block of content) {
      if (typeof block !== "object" || block === null) continue;
      const b = block as Record<string, unknown>;
      if (b["type"] !== "tool_result") continue;
      const tcId = String(b["tool_use_id"] ?? "");
      const entry = _pendingToolCalls.get(tcId);
      if (!entry) continue;
      _pendingToolCalls.delete(tcId);
      latencies.push({ tool_call_id: tcId, name: entry.name, latency: now - entry.ts });
    }
  }
  return latencies.length > 0 ? latencies : null;
}

// ---------------------------------------------------------------------------
// Emit helpers
// ---------------------------------------------------------------------------

async function _emitMessagesTrace(
  params: Record<string, unknown>,
  response: Record<string, unknown>,
  start: number,
  end: number,
  toolCallLatencies: unknown
): Promise<void> {
  const content = response["content"];
  const usage = response["usage"];
  _recordDispatchedToolCalls(content);
  const text = _extractTextFromContent(content);
  const toolUses = _extractToolUses(content);
  const thinking = _extractThinking(content);
  const tokens = _extractUsage(response);

  await logTrace(
    toPlainObject({
      type: "llm",
      integration: TraceType.Anthropic,
      model: (params["model"] as string) ?? (response["model"] as string) ?? null,
      messages: _toJsonable(params["messages"]) as unknown[],
      system: _toJsonable(params["system"]),
      tools: _toJsonable(params["tools"]),
      tool_choice: _toJsonable(params["tool_choice"]),
      response: text,
      thinking,
      tool_uses: toolUses,
      tool_call_latencies: toolCallLatencies,
      mcp_servers: _extractMcpServers(params),
      mcp_calls: _extractMcpBlocks(content),
      mcp_results: _extractMcpResultsFromMessages(params["messages"]),
      stop_reason: (response["stop_reason"] as string) ?? null,
      latency: end - start,
      parent_id: currentParentId(),
      tokens: tokens
        ? { prompt: tokens.prompt, completion: tokens.completion, total: tokens.total }
        : null,
      prompt_cache_read_tokens: _cacheReadTokens(usage),
      prompt_cache_creation_tokens: _cacheCreationTokens(usage),
    })
  );
}

async function _emitMessagesError(
  params: Record<string, unknown>,
  error: Error,
  start: number,
  end: number,
  api?: string
): Promise<void> {
  await logTrace(
    toPlainObject({
      type: "llm",
      integration: TraceType.Anthropic,
      api: api ?? null,
      model: params["model"] as string,
      messages: _toJsonable(params["messages"]) as unknown[],
      system: _toJsonable(params["system"]),
      tools: _toJsonable(params["tools"]),
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
  end: number
): Promise<void> {
  const attackTypes = exc.attackTypes;
  const has = (t: string) => attackTypes.includes(t);
  await logTrace({
    type: "llm",
    integration: TraceType.Anthropic,
    api: "messages",
    model: params["model"],
    messages: _toJsonable(params["messages"]),
    system: _toJsonable(params["system"]),
    tools: _toJsonable(params["tools"]),
    latency: end - start,
    parent_id: currentParentId(),
    success: false,
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
// Generic prototype patcher
// ---------------------------------------------------------------------------

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyFn = (this: unknown, ...args: any[]) => unknown;

/** Patch ClassName.prototype[method] from the first resolvable module path. */
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
      continue;
    }
    const cls = mod[className] as { prototype?: Record<string, unknown> } | undefined;
    if (!cls || !cls.prototype) continue;
    const orig = cls.prototype[method];
    if (typeof orig !== "function") return;
    if ((orig as unknown as Record<string, boolean>)["_fluiq_patched"]) return;
    const wrapped = wrap(orig as AnyFn);
    (wrapped as unknown as Record<string, boolean>)["_fluiq_patched"] = true;
    cls.prototype[method] = wrapped;
    return;
  }
}

// ---------------------------------------------------------------------------
// Patch: messages.create
// ---------------------------------------------------------------------------

/** Result of the traced pipeline: the value to resolve, plus the raw HTTP
 *  response (when there was a real network call) so we can honour the SDK's
 *  APIPromise contract (`.withResponse()` / `.asResponse()`). */
interface TracedCreate {
  result: unknown;
  rawResponse: unknown;
}

async function _runTracedMessagesCreate(
  original: AnyFn,
  self: unknown,
  params: Record<string, unknown>,
  opts: unknown
): Promise<TracedCreate> {
  _gcPendingToolCalls();
  const toolCallLatencies = _computeToolCallLatencies(
    params["messages"] as unknown[] | undefined
  );
  const traceId = randomUUID();
  const start = Date.now() / 1000;

  await logTrace(
    toPlainObject({
      type: "llm",
      integration: TraceType.Anthropic,
      api: "messages",
      trace_id: traceId,
      model: params["model"] as string,
      messages: _toJsonable(params["messages"]) as unknown[],
      system: _toJsonable(params["system"]),
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
      learnFromAnthropicMessages(params["messages"]);
    } catch {
      /* tool-cache learning must never break the call */
    }
    try {
      maybeInjectAnthropicCacheControl(params);
    } catch {
      /* cache_control injection is best-effort */
    }

    const cached = await preCallOptimize(params, "anthropic");
    if (cached != null) {
      const end = Date.now() / 1000;
      await logTrace({
        type: "llm",
        integration: TraceType.Anthropic,
        api: "messages",
        trace_id: traceId,
        model: params["model"],
        messages: _toJsonable(params["messages"]),
        system: _toJsonable(params["system"]),
        tools: _toJsonable(params["tools"]),
        response: cached["response"],
        tool_uses: cached["tool_uses"],
        mcp_calls: cached["mcp_calls"],
        latency: end - start,
        parent_id: currentParentId(),
        _cache_hit: true,
        tokens: null,
      });
      return { result: _buildAnthropicCachedResponse(cached, params), rawResponse: null };
    }

    // Call the original create. It returns an APIPromise; use `.withResponse()`
    // so we can thread the raw HTTP response back out (the SDK's MessageStream
    // helper calls `messages.create(...).withResponse()` and needs it).
    let response: unknown;
    let rawResponse: unknown = null;
    try {
      const apiPromise = original.call(self, params, opts);
      if (apiPromise && typeof (apiPromise as { withResponse?: unknown }).withResponse === "function") {
        const wr = await (apiPromise as { withResponse(): Promise<{ data: unknown; response: unknown }> }).withResponse();
        response = wr.data;
        rawResponse = wr.response;
      } else {
        response = await apiPromise;
      }
    } catch (e) {
      await _emitMessagesError(params, e as Error, start, Date.now() / 1000);
      throw e;
    }

    if (params["stream"]) {
      return {
        result: _wrapAnthropicStream(response, params, start, traceId, toolCallLatencies),
        rawResponse,
      };
    }

    const end = Date.now() / 1000;
    await _emitMessagesTrace(params, response as Record<string, unknown>, start, end, toolCallLatencies);
    return { result: response, rawResponse };
  });
}

function _buildMessagesCreateWrapper(original: AnyFn): AnyFn {
  // NOTE: intentionally NOT `async`. The Anthropic SDK's `messages.create()`
  // returns an `APIPromise` and its `messages.stream()` helper calls
  // `messages.create(...).withResponse()` on the return value. An `async`
  // wrapper would resolve to a plain Promise and strip `.withResponse()`.
  // Instead we return an APIPromise-like thenable that runs the traced pipeline
  // once and exposes `then`/`withResponse`/`asResponse`.
  return function (this: unknown, params: Record<string, unknown>, opts?: unknown) {
    if (isInLangchainLlm()) return original.call(this, params, opts);
    const self = this;
    // Kick off the traced pipeline eagerly (matches APIPromise, which starts the
    // request on creation). Shared so `then` and `withResponse` don't double-run.
    const exec = runInRootContext(() => _runTracedMessagesCreate(original, self, params, opts));
    const resultOf = () => exec.then((x) => x.result);
    return {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      then: (onF: any, onR: any) => resultOf().then(onF, onR),
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      catch: (onR: any) => resultOf().catch(onR),
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      finally: (onFin: any) => resultOf().finally(onFin),
      withResponse: () =>
        exec.then((x) => ({ data: x.result, response: x.rawResponse, request_id: null })),
      asResponse: () => exec.then((x) => x.rawResponse),
    };
  };
}

// ---------------------------------------------------------------------------
// messages.stream() helper
// ---------------------------------------------------------------------------
// Intentionally NOT patched. `messages.stream()` returns a `MessageStream`
// whose `_createMessage` calls `messages.create({ stream: true })` internally —
// which is already patched above. Patching the helper too would double-trace,
// and wrapping the returned MessageStream breaks its ES private fields
// (`.on()` / `.finalMessage()`). Letting the inner `create` handle it yields a
// single trace and keeps the MessageStream API fully intact.

// ---------------------------------------------------------------------------
// Patch: messages.countTokens
// ---------------------------------------------------------------------------

async function _emitCountTokensTrace(
  api: string,
  params: Record<string, unknown>,
  response: unknown,
  start: number,
  end: number
): Promise<void> {
  const inputTokens = response && typeof response === "object"
    ? (response as Record<string, unknown>)["input_tokens"]
    : undefined;
  await logTrace(
    toPlainObject({
      type: "llm",
      integration: TraceType.Anthropic,
      api,
      model: params["model"] as string,
      messages: _toJsonable(params["messages"]),
      system: _toJsonable(params["system"]),
      tools: _toJsonable(params["tools"]),
      response: _toJsonable(response),
      latency: end - start,
      parent_id: currentParentId(),
      tokens:
        typeof inputTokens === "number"
          ? { prompt: inputTokens, completion: null, total: inputTokens }
          : null,
    })
  );
}

function _buildCountTokensWrapper(api: string): (orig: AnyFn) => AnyFn {
  return (original) =>
    async function (this: unknown, params: Record<string, unknown>, opts?: unknown) {
      if (isInLangchainLlm()) return original.call(this, params, opts);
      const start = Date.now() / 1000;
      let response: unknown;
      try {
        response = await original.call(this, params, opts);
      } catch (e) {
        await _emitMessagesError(params, e as Error, start, Date.now() / 1000, api);
        throw e;
      }
      await _emitCountTokensTrace(api, params, response, start, Date.now() / 1000);
      return response;
    };
}

// ---------------------------------------------------------------------------
// Public patch entry points
// ---------------------------------------------------------------------------

const _MESSAGES_PATHS = [
  "@anthropic-ai/sdk/resources/messages/messages",
  "@anthropic-ai/sdk/resources/messages",
];
const _BETA_MESSAGES_PATHS = [
  "@anthropic-ai/sdk/resources/beta/messages/messages",
  "@anthropic-ai/sdk/resources/beta/messages",
];

export function patchAnthropic(): void {
  _patchProto(_MESSAGES_PATHS, "Messages", "create", _buildMessagesCreateWrapper);
  _patchProto(_MESSAGES_PATHS, "Messages", "countTokens", _buildCountTokensWrapper("messages.count_tokens"));
}

export function patchAnthropicBeta(): void {
  _patchProto(_BETA_MESSAGES_PATHS, "Messages", "create", _buildMessagesCreateWrapper);
  _patchProto(
    _BETA_MESSAGES_PATHS,
    "Messages",
    "countTokens",
    _buildCountTokensWrapper("beta.messages.count_tokens")
  );
}

// ---------------------------------------------------------------------------
// Streaming wrapper
// ---------------------------------------------------------------------------

function _wrapAnthropicStream(
  stream: unknown,
  params: Record<string, unknown>,
  start: number,
  traceId: string,
  toolCallLatencies: unknown
): AsyncIterable<unknown> {
  const originalStream = stream as AsyncIterable<unknown>;
  const chunks: unknown[] = [];
  const needsGate = _config.secure && _config.secure_mode === "block";

  async function emitEnd(): Promise<void> {
    const end = Date.now() / 1000;
    const accumulated = _accumulateAnthropicChunks(chunks);
    await withLlmTraceId(traceId, () =>
      logTrace(
        toPlainObject({
          type: "llm",
          integration: TraceType.Anthropic,
          api: "messages.stream",
          trace_id: traceId,
          model: (params["model"] as string) ?? accumulated.model,
          messages: _toJsonable(params["messages"]) as unknown[],
          system: _toJsonable(params["system"]),
          tools: _toJsonable(params["tools"]),
          response: accumulated.text,
          thinking: accumulated.thinking,
          tool_uses: accumulated.toolUses,
          tool_call_latencies: toolCallLatencies,
          mcp_servers: _extractMcpServers(params),
          mcp_results: _extractMcpResultsFromMessages(params["messages"]),
          stop_reason: accumulated.stopReason,
          latency: end - start,
          parent_id: currentParentId(),
          tokens: accumulated.usage as unknown as import("./shared/models").Tokens | null,
          prompt_cache_read_tokens: accumulated.cacheReadTokens,
          prompt_cache_creation_tokens: accumulated.cacheCreationTokens,
        })
      )
    );
  }

  async function* bufferedGenerator(): AsyncGenerator<unknown> {
    try {
      for await (const event of originalStream) chunks.push(event);
    } catch (e) {
      await _emitMessagesError(params, e as Error, start, Date.now() / 1000, "messages.stream");
      throw e;
    }
    await emitEnd(); // may throw FluiqSecurityError via the response gate
    for (const event of chunks) yield event;
  }

  async function* generator(): AsyncGenerator<unknown> {
    try {
      for await (const event of originalStream) {
        chunks.push(event);
        yield event;
      }
    } catch (e) {
      await _emitMessagesError(params, e as Error, start, Date.now() / 1000, "messages.stream");
      throw e;
    }
    await emitEnd();
  }

  // Preserve the original Stream object (notably `.controller`, read by
  // MessageStream._createMessage) and only override iteration. A Proxy keeps
  // method `this`-binding intact so the SDK's ES private fields keep working —
  // unlike Object.create / a bare generator.
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

interface AnthropicAccumulated {
  text: string | null;
  model: string | null;
  thinking: unknown[] | null;
  toolUses: unknown[] | null;
  stopReason: string | null;
  usage: Record<string, number | null> | null;
  cacheReadTokens: number | null;
  cacheCreationTokens: number | null;
}

function _accumulateAnthropicChunks(chunks: unknown[]): AnthropicAccumulated {
  let text = "";
  let model: string | null = null;
  let stopReason: string | null = null;
  const thinkingParts: string[] = [];
  const toolUsesMap: Record<number, Record<string, unknown>> = {};
  let usage: Record<string, number | null> | null = null;
  let cacheReadTokens: number | null = null;
  let cacheCreationTokens: number | null = null;

  for (const event of chunks) {
    const e = event as Record<string, unknown>;
    const type = e["type"] as string | undefined;

    if (type === "message_start") {
      const msg = e["message"] as Record<string, unknown> | undefined;
      if (msg?.["model"]) model = String(msg["model"]);
      const u = msg?.["usage"] as Record<string, unknown> | undefined;
      if (u) {
        const inp = Number(u["input_tokens"] ?? 0);
        usage = { prompt: inp, completion: 0, total: inp };
        const cr = _cacheReadTokens(u);
        const cc = _cacheCreationTokens(u);
        if (cr != null) cacheReadTokens = cr;
        if (cc != null) cacheCreationTokens = cc;
      }
    }

    if (type === "content_block_start") {
      const block = e["content_block"] as Record<string, unknown> | undefined;
      const idx = Number(e["index"] ?? 0);
      if (block?.["type"] === "tool_use") {
        toolUsesMap[idx] = {
          type: "tool_use",
          id: block["id"],
          name: block["name"],
          input: "",
        };
      }
    }

    if (type === "content_block_delta") {
      const delta = e["delta"] as Record<string, unknown> | undefined;
      const idx = Number(e["index"] ?? 0);
      if (delta?.["type"] === "text_delta") text += String(delta["text"] ?? "");
      if (delta?.["type"] === "thinking_delta") thinkingParts.push(String(delta["thinking"] ?? ""));
      if (delta?.["type"] === "input_json_delta") {
        const tu = toolUsesMap[idx];
        if (tu) tu["input"] = String(tu["input"] ?? "") + String(delta["partial_json"] ?? "");
      }
    }

    if (type === "message_delta") {
      const delta = e["delta"] as Record<string, unknown> | undefined;
      if (delta?.["stop_reason"]) stopReason = String(delta["stop_reason"]);
      const u = e["usage"] as Record<string, unknown> | undefined;
      if (u && usage) usage["completion"] = Number(u["output_tokens"] ?? 0);
      if (u && usage) usage["total"] = (usage["prompt"] ?? 0) + (usage["completion"] ?? 0);
    }
  }

  // Parse accumulated tool_use input JSON into structured objects.
  const toolUsesArr = Object.values(toolUsesMap).map((tu) => {
    const raw = String(tu["input"] ?? "");
    let parsed: unknown = {};
    if (raw) {
      try {
        parsed = JSON.parse(raw);
      } catch {
        parsed = { _raw: raw };
      }
    }
    return { type: "tool_use", id: tu["id"], name: tu["name"], input: parsed };
  });

  return {
    text: text || null,
    model,
    thinking: thinkingParts.length > 0 ? [{ type: "thinking", thinking: thinkingParts.join("") }] : null,
    toolUses: toolUsesArr.length > 0 ? toolUsesArr : null,
    stopReason,
    usage,
    cacheReadTokens,
    cacheCreationTokens,
  };
}

function _buildAnthropicCachedResponse(
  payload: Record<string, unknown>,
  params: Record<string, unknown>
): Record<string, unknown> {
  const text = payload["response"] as string | null;
  const toolUses = payload["tool_uses"] as unknown[] | null;

  const contentBlocks: unknown[] = [];
  if (text) contentBlocks.push({ type: "text", text });
  for (const tu of toolUses ?? []) {
    const t = tu as Record<string, unknown>;
    contentBlocks.push({ type: "tool_use", id: t["id"], name: t["name"], input: t["input"] });
  }
  if (contentBlocks.length === 0) contentBlocks.push({ type: "text", text: "" });

  return {
    id: "fluiq-cached",
    type: "message",
    role: "assistant",
    content: contentBlocks,
    model: params["model"] ?? "",
    stop_reason: toolUses ? "tool_use" : "end_turn",
    stop_sequence: null,
    // Served from cache — no provider call was made, so all token counts are 0.
    // Use a zeroed object (not null) so caller code that reads e.g.
    // `msg.usage.cache_read_input_tokens` doesn't throw on a cache hit.
    usage: {
      input_tokens: 0,
      output_tokens: 0,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 0,
    },
    _fluiq_cached: true,
  };
}
