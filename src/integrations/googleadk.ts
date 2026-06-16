/**
 * Google ADK integration (@google/adk, the JS port of google.adk).
 *
 * Provides FluiqADKPlugin and patchGoogleADK(), which registers a single shared
 * instance into every PluginManager. Silently skipped if @google/adk is not
 * installed.
 *
 * Why this differs from the Python plugin: ADK-JS v1.x only invokes a subset of
 * the BasePlugin callbacks at runtime. `runBeforeAgentCallback`/
 * `runAfterAgentCallback` are defined on PluginManager but never called by the
 * agent run loop, so agent spans cannot come from those hooks. ADK also bundles
 * its own `@google/genai`, so the SDK's Gemini patch never sees ADK's model
 * calls. We therefore emit everything from the callbacks that DO fire — model
 * and tool callbacks — and reconstruct the agent tree by walking
 * `invocationContext.agent.parentAgent`: the first model/tool activity under an
 * agent lazily opens a span for it and every ancestor (top-down, so parent_ids
 * link correctly), and all open agent spans for an invocation are closed in
 * afterRunCallback.
 */
import { randomUUID } from "crypto";
import { logTrace } from "../tracer";
import { TraceType, toPlainObject, Tokens } from "./shared/models";
import { currentParentId } from "./shared/context";

let _registered = false;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function _toJsonable(obj: unknown): unknown {
  if (obj == null) return null;
  if (typeof obj !== "object") return obj;
  const maybeToJSON = (obj as { toJSON?: () => unknown }).toJSON;
  if (typeof maybeToJSON === "function") {
    try {
      return _toJsonable(maybeToJSON.call(obj));
    } catch {
      // fall through to structural walk
    }
  }
  if (Array.isArray(obj)) return obj.map(_toJsonable);
  const o = obj as Record<string, unknown>;
  try {
    return Object.fromEntries(Object.entries(o).map(([k, v]) => [k, _toJsonable(v)]));
  } catch {
    return String(obj);
  }
}

function _contentToText(content: unknown): string | null {
  if (!content) return null;
  const c = content as Record<string, unknown>;
  const parts = c["parts"];
  if (!Array.isArray(parts)) return null;
  const texts: string[] = [];
  for (const part of parts) {
    const p = part as Record<string, unknown>;
    if (typeof p["text"] === "string" && p["text"]) texts.push(p["text"]);
  }
  return texts.join("\n") || null;
}

/** Extracts functionCall parts from a Content so tool-invocation turns aren't blank. */
function _functionCalls(content: unknown): unknown[] | null {
  const parts = _g(content, "parts");
  if (!Array.isArray(parts)) return null;
  const calls: unknown[] = [];
  for (const part of parts) {
    const fc = _g(part, "functionCall", "function_call");
    if (fc) calls.push(_toJsonable(fc));
  }
  return calls.length ? calls : null;
}

// ADK reuses the `errorCode` field to carry the stop/finish signal (e.g.
// "STOP", "MAX_TOKENS") on otherwise-successful turns — notably the empty
// final turn emitted when a LoopAgent escalates. Only a code outside this set
// is a real failure.
const _NON_ERROR_CODES = new Set(["STOP", "MAX_TOKENS", "FINISH_REASON_UNSPECIFIED"]);

function _modelSucceeded(llmResponse: unknown): boolean {
  const code = _g(llmResponse, "errorCode", "error_code");
  return code == null || _NON_ERROR_CODES.has(String(code));
}

function _g(obj: unknown, ...keys: string[]): unknown {
  if (!obj || typeof obj !== "object") return undefined;
  const o = obj as Record<string, unknown>;
  for (const k of keys) {
    if (o[k] !== undefined) return o[k];
  }
  return undefined;
}

function _invocationContext(callbackContext: unknown): Record<string, unknown> | null {
  const ctx = callbackContext as Record<string, unknown> | null;
  if (!ctx || typeof ctx !== "object") return null;
  try {
    const ic = ctx["invocationContext"] ?? ctx["_invocation_context"];
    return ic && typeof ic === "object" ? (ic as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

function _agentName(agent: unknown): string | null {
  if (!agent || typeof agent !== "object") return null;
  const name = (agent as Record<string, unknown>)["name"];
  return name ? String(name) : null;
}

function _agentModel(agent: unknown): string | null {
  if (!agent || typeof agent !== "object") return null;
  const model = (agent as Record<string, unknown>)["model"];
  if (model == null) return null;
  if (typeof model === "string") return model;
  const m = model as Record<string, unknown>;
  const name = m["model"] ?? m["name"];
  return name ? String(name) : null;
}

function _userText(ic: Record<string, unknown> | null): string | null {
  if (!ic) return null;
  try {
    return _contentToText(ic["userContent"] ?? ic["user_content"]);
  } catch {
    return null;
  }
}

function _systemInstruction(llmRequest: unknown): unknown {
  const config = _g(llmRequest, "config");
  return _toJsonable(_g(config, "systemInstruction", "system_instruction") ?? null);
}

function _extractTokens(llmResponse: unknown): Tokens | null {
  const usage = _g(llmResponse, "usageMetadata", "usage_metadata");
  if (!usage) return null;
  const prompt = _g(usage, "promptTokenCount", "prompt_token_count");
  const completion = _g(usage, "candidatesTokenCount", "candidates_token_count");
  const total = _g(usage, "totalTokenCount", "total_token_count");
  return {
    prompt: prompt != null ? Number(prompt) : null,
    completion: completion != null ? Number(completion) : null,
    total: total != null ? Number(total) : null,
  };
}

function _toolInputSchema(tool: unknown): unknown {
  if (!tool || typeof tool !== "object") return null;
  const t = tool as Record<string, unknown>;
  for (const attr of ["_getDeclaration", "getDeclaration"]) {
    const fn = t[attr];
    if (typeof fn !== "function") continue;
    try {
      const decl = (fn as () => unknown).call(tool);
      if (decl && typeof decl === "object") {
        return _toJsonable((decl as Record<string, unknown>)["parameters"] ?? null);
      }
    } catch {
      // some tools throw when no declaration is available
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// FluiqADKPlugin
// ---------------------------------------------------------------------------

interface AgentEntry {
  traceId: string;
  parentId: string | null;
  start: number;
  lastActivityEnd: number;
  agentName: string | null;
  model: string | null;
  input: string | null;
  invocationId: string | null;
  lastResponse?: string;
}

interface LlmEntry {
  traceId: string;
  parentId: string | null;
  start: number;
  model: string | null;
}

interface ToolEntry {
  traceId: string;
  parentId: string | null;
  start: number;
  name: string | null;
  description: string | null;
  input: unknown;
  schema: unknown;
  agentKey: string;
}

export class FluiqADKPlugin {
  /** ADK identifies plugins by name; must be unique within a PluginManager. */
  readonly name = "fluiq";

  private _agents = new Map<string, AgentEntry>();
  // One in-flight model call per (invocation, agent); ADK serializes them.
  private _llms = new Map<string, LlmEntry>();
  // Keyed by the live toolContext object (one per tool invocation).
  private _tools = new WeakMap<object, ToolEntry>();

  // Serialize emissions: fire-and-forget POSTs can otherwise be delivered out
  // of order, stranding a "running" row that its completion raced past.
  private _emitChain: Promise<void> = Promise.resolve();

  private _emit(fields: Record<string, unknown>): void {
    let payload: Record<string, unknown>;
    try {
      payload = toPlainObject({ integration: TraceType.GoogleADK, ...fields });
    } catch {
      return; // fail open
    }
    this._emitChain = this._emitChain.then(() => logTrace(payload).catch(() => {}));
  }

  private _emitStart(fields: Record<string, unknown>): void {
    this._emit({ status: "running", started_at: Date.now() / 1000, ...fields });
  }

  private static _key(invocationId: string | null, agentName: string | null): string {
    return `${invocationId ?? ""}:${agentName ?? ""}`;
  }

  /**
   * Ensures an agent span is open for `ic.agent` and every ancestor, creating
   * them root-first so parent_ids resolve. Returns the leaf agent's entry.
   */
  private _ensureAgentChain(ic: Record<string, unknown> | null): AgentEntry | null {
    if (!ic) return null;
    const invocationId = ic["invocationId"] != null ? String(ic["invocationId"]) : null;
    const userInput = _userText(ic);

    // Walk parentAgent up to the root, then reverse to create top-down.
    const chain: unknown[] = [];
    let a: unknown = ic["agent"];
    const seen = new Set<unknown>();
    while (a && typeof a === "object" && !seen.has(a)) {
      seen.add(a);
      chain.push(a);
      a = (a as Record<string, unknown>)["parentAgent"] ?? (a as Record<string, unknown>)["parent_agent"];
    }
    chain.reverse();

    let parentTrace: string | null = currentParentId();
    let leaf: AgentEntry | null = null;
    for (const agent of chain) {
      const agentName = _agentName(agent);
      const key = FluiqADKPlugin._key(invocationId, agentName);
      let entry = this._agents.get(key);
      if (!entry) {
        const now = Date.now() / 1000;
        entry = {
          traceId: randomUUID(),
          parentId: parentTrace,
          start: now,
          lastActivityEnd: now,
          agentName,
          model: _agentModel(agent),
          input: userInput,
          invocationId,
        };
        this._agents.set(key, entry);
        this._emitStart({
          type: "agent",
          function: agentName,
          model: entry.model,
          input: userInput,
          trace_id: entry.traceId,
          parent_id: parentTrace,
          invocation_id: invocationId,
        });
      }
      parentTrace = entry.traceId;
      leaf = entry;
    }
    return leaf;
  }

  /** Bumps lastActivityEnd on the leaf agent and all its ancestors. */
  private _touchChain(ic: Record<string, unknown> | null, end: number): void {
    if (!ic) return;
    const invocationId = ic["invocationId"] != null ? String(ic["invocationId"]) : null;
    let a: unknown = ic["agent"];
    const seen = new Set<unknown>();
    while (a && typeof a === "object" && !seen.has(a)) {
      seen.add(a);
      const entry = this._agents.get(FluiqADKPlugin._key(invocationId, _agentName(a)));
      if (entry && end > entry.lastActivityEnd) entry.lastActivityEnd = end;
      a = (a as Record<string, unknown>)["parentAgent"] ?? (a as Record<string, unknown>)["parent_agent"];
    }
  }

  // --- model spans (these callbacks DO fire) ------------------------------

  async beforeModelCallback(opts: { callbackContext?: unknown; llmRequest?: unknown }): Promise<undefined> {
    const ic = _invocationContext(opts.callbackContext);
    const leaf = this._ensureAgentChain(ic);
    if (!ic || !leaf) return undefined;
    const invocationId = ic["invocationId"] != null ? String(ic["invocationId"]) : null;
    const agentName = _agentName(ic["agent"]);
    const key = FluiqADKPlugin._key(invocationId, agentName);

    const llmRequest = (opts.llmRequest ?? {}) as Record<string, unknown>;
    const model = (_g(llmRequest, "model") as string | null) ?? leaf.model;
    const traceId = randomUUID();
    this._llms.set(key, { traceId, parentId: leaf.traceId, start: Date.now() / 1000, model });

    this._emitStart({
      type: "llm",
      model,
      contents: _toJsonable(llmRequest["contents"] ?? null),
      system_instruction: _systemInstruction(llmRequest),
      trace_id: traceId,
      parent_id: leaf.traceId,
      invocation_id: invocationId,
    });
    return undefined;
  }

  async afterModelCallback(opts: { callbackContext?: unknown; llmResponse?: unknown }): Promise<undefined> {
    const ic = _invocationContext(opts.callbackContext);
    if (!ic) return undefined;
    const invocationId = ic["invocationId"] != null ? String(ic["invocationId"]) : null;
    const agentName = _agentName(ic["agent"]);
    const key = FluiqADKPlugin._key(invocationId, agentName);
    const llm = this._llms.get(key);
    if (!llm) return undefined;
    this._llms.delete(key);

    const llmResponse = (opts.llmResponse ?? {}) as Record<string, unknown>;
    const text = _contentToText(llmResponse["content"]);
    const finish = _g(llmResponse, "finishReason", "finish_reason");
    const end = Date.now() / 1000;

    this._emit({
      type: "llm",
      model: llm.model,
      response: text,
      function_calls: _functionCalls(llmResponse["content"]),
      tokens: _extractTokens(llmResponse),
      finish_reasons: finish != null ? [String(finish)] : null,
      latency: end - llm.start,
      trace_id: llm.traceId,
      parent_id: llm.parentId,
      invocation_id: invocationId,
      success: _modelSucceeded(llmResponse),
    });

    const agent = this._agents.get(key);
    if (agent && text) agent.lastResponse = text;
    this._touchChain(ic, end);
    return undefined;
  }

  async onModelErrorCallback(opts: {
    callbackContext?: unknown;
    llmRequest?: unknown;
    error?: unknown;
  }): Promise<undefined> {
    const ic = _invocationContext(opts.callbackContext);
    const invocationId = ic && ic["invocationId"] != null ? String(ic["invocationId"]) : null;
    const agentName = ic ? _agentName(ic["agent"]) : null;
    const key = FluiqADKPlugin._key(invocationId, agentName);
    const llm = this._llms.get(key);
    if (llm) this._llms.delete(key);
    const parentId = llm ? llm.parentId : this._agents.get(key)?.traceId ?? currentParentId();
    const end = Date.now() / 1000;

    this._emit({
      type: "llm",
      model: llm?.model ?? (_g(opts.llmRequest, "model") as string | null) ?? null,
      output: String(opts.error),
      latency: llm ? end - llm.start : undefined,
      trace_id: llm?.traceId ?? randomUUID(),
      parent_id: parentId,
      invocation_id: invocationId,
      success: false,
    });
    this._touchChain(ic, end);
    return undefined;
  }

  // --- tool spans (these callbacks DO fire) -------------------------------

  async beforeToolCallback(opts: {
    tool?: unknown;
    toolArgs?: unknown;
    toolContext?: unknown;
  }): Promise<undefined> {
    const { tool, toolArgs, toolContext } = opts;
    if (!toolContext || typeof toolContext !== "object") return undefined;
    const ic = _invocationContext(toolContext);
    const leaf = this._ensureAgentChain(ic);
    const invocationId = ic && ic["invocationId"] != null ? String(ic["invocationId"]) : null;
    const agentName = ic ? _agentName(ic["agent"]) : null;
    const parentId = leaf ? leaf.traceId : currentParentId();

    const traceId = randomUUID();
    const toolName = tool && typeof tool === "object"
      ? (((tool as Record<string, unknown>)["name"] ?? null) as string | null)
      : null;
    const toolInput = _toJsonable(toolArgs);

    this._tools.set(toolContext as object, {
      traceId,
      parentId,
      start: Date.now() / 1000,
      name: toolName,
      description: tool && typeof tool === "object"
        ? (((tool as Record<string, unknown>)["description"] ?? null) as string | null)
        : null,
      input: toolInput,
      schema: _toolInputSchema(tool),
      agentKey: FluiqADKPlugin._key(invocationId, agentName),
    });

    this._emitStart({
      type: "tool",
      function: toolName,
      input: toolInput,
      trace_id: traceId,
      parent_id: parentId,
      invocation_id: invocationId,
    });
    return undefined;
  }

  async afterToolCallback(opts: {
    tool?: unknown;
    toolArgs?: unknown;
    toolContext?: unknown;
    result?: unknown;
  }): Promise<undefined> {
    const { toolContext, result } = opts;
    if (!toolContext || typeof toolContext !== "object") return undefined;
    const state = this._tools.get(toolContext as object);
    if (!state) return undefined;
    this._tools.delete(toolContext as object);

    const end = Date.now() / 1000;
    this._emit({
      type: "tool",
      function: state.name,
      input: state.input,
      output: _toJsonable(result),
      tools: state.name
        ? [{ name: state.name, description: state.description, input_schema: state.schema }]
        : null,
      latency: end - state.start,
      trace_id: state.traceId,
      parent_id: state.parentId,
      success: true,
    });
    this._touchChain(_invocationContext(toolContext), end);
    return undefined;
  }

  async onToolErrorCallback(opts: {
    tool?: unknown;
    toolArgs?: unknown;
    toolContext?: unknown;
    error?: unknown;
  }): Promise<undefined> {
    const { toolContext, error } = opts;
    if (!toolContext || typeof toolContext !== "object") return undefined;
    const state = this._tools.get(toolContext as object);
    if (!state) return undefined;
    this._tools.delete(toolContext as object);

    const end = Date.now() / 1000;
    this._emit({
      type: "tool",
      function: state.name,
      input: state.input,
      output: String(error),
      latency: end - state.start,
      trace_id: state.traceId,
      parent_id: state.parentId,
      success: false,
    });
    this._touchChain(_invocationContext(toolContext), end);
    return undefined;
  }

  // --- run lifecycle: close all agent spans for the invocation ------------

  async afterRunCallback(opts: { invocationContext?: unknown }): Promise<undefined> {
    const ic = opts.invocationContext as Record<string, unknown> | undefined;
    const invocationId = ic && ic["invocationId"] != null ? String(ic["invocationId"]) : null;
    if (!invocationId) return undefined;

    for (const [key, entry] of [...this._agents.entries()]) {
      if (entry.invocationId !== invocationId) continue;
      this._agents.delete(key);
      const end = Math.max(entry.lastActivityEnd, entry.start);
      this._emit({
        type: "agent",
        function: entry.agentName,
        model: entry.model,
        input: entry.input,
        output: entry.lastResponse ?? null,
        latency: end - entry.start,
        trace_id: entry.traceId,
        parent_id: entry.parentId,
        invocation_id: entry.invocationId,
        success: true,
      });
    }
    return undefined;
  }

  // --- no-op callbacks ----------------------------------------------------
  // PluginManager.runCallbacks invokes every callback method unconditionally
  // and rethrows on error, so each hook ADK may call must exist and resolve to
  // `undefined` (which tells ADK to proceed without modification).

  async onUserMessageCallback(): Promise<undefined> {
    return undefined;
  }
  async beforeRunCallback(): Promise<undefined> {
    return undefined;
  }
  async onEventCallback(): Promise<undefined> {
    return undefined;
  }
  async beforeAgentCallback(): Promise<undefined> {
    return undefined;
  }
  async afterAgentCallback(): Promise<undefined> {
    return undefined;
  }
  async beforeToolSelection(): Promise<undefined> {
    return undefined;
  }
  async beforeContextCompaction(): Promise<undefined> {
    return undefined;
  }
  async afterContextCompaction(): Promise<undefined> {
    return undefined;
  }
}

// ---------------------------------------------------------------------------
// Public entry point — registers FluiqADKPlugin into every PluginManager
// ---------------------------------------------------------------------------

export function patchGoogleADK(): void {
  if (_registered) return;

  // Runner constructs `new PluginManager(...)` from an internal import, not the
  // package root export, so replacing the root export would never reach it.
  // Both resolve to the same class object (single module instance), so we patch
  // the shared prototype's earliest lifecycle hooks to lazily register our
  // singleton plugin into each manager the first time it runs.
  const mod = require("@google/adk") as {
    PluginManager?: { prototype: Record<string, unknown> };
  };
  const PluginManager = mod.PluginManager;
  if (!PluginManager || !PluginManager.prototype) return;

  const proto = PluginManager.prototype as Record<string, unknown>;
  const singleton = new FluiqADKPlugin();
  const registeredManagers = new WeakSet<object>();

  const ensureRegistered = (self: unknown): void => {
    if (!self || typeof self !== "object") return;
    try {
      if (registeredManagers.has(self as object)) return;
      registeredManagers.add(self as object);
      const mgr = self as {
        getPlugin?: (name: string) => unknown;
        registerPlugin?: (plugin: unknown) => void;
      };
      if (typeof mgr.getPlugin === "function" && mgr.getPlugin("fluiq")) return;
      if (typeof mgr.registerPlugin === "function") mgr.registerPlugin(singleton);
    } catch {
      // already registered / incompatible manager — leave it alone
    }
  };

  for (const hook of ["runOnUserMessageCallback", "runBeforeRunCallback"]) {
    const original = proto[hook] as
      | (((...args: unknown[]) => unknown) & { _fluiqPatched?: boolean })
      | undefined;
    if (typeof original !== "function" || original._fluiqPatched) continue;
    const patched = function (this: unknown, ...args: unknown[]) {
      ensureRegistered(this);
      return original.apply(this, args);
    } as ((...args: unknown[]) => unknown) & { _fluiqPatched?: boolean };
    patched._fluiqPatched = true;
    proto[hook] = patched;
  }

  _registered = true;
}
