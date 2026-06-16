/**
 * LangChain + LangGraph integration.
 *
 * Provides FluiqCallbackHandler (extends BaseCallbackHandler) for tracing
 * LLM calls, chains, and tool invocations. Auto-registers via
 * register_configure_hook so all LangChain/LangGraph runs are captured
 * automatically after patchLangchain() is called.
 *
 * Also used by patchLangGraph (same handler, different integration label).
 */
import { randomUUID } from "crypto";
import { logTrace } from "../tracer";
import { TraceType, toPlainObject } from "./shared/models";
import {
  currentParentId,
  enterLangchainLlm,
  exitLangchainLlm,
} from "./shared/context";
import { preCallGuard } from "./shared/securityGate";

// ---------------------------------------------------------------------------
// LangGraph metadata extraction
// ---------------------------------------------------------------------------

const LANGGRAPH_META_KEYS = [
  "langgraph_node",
  "langgraph_step",
  "langgraph_path",
  "langgraph_triggers",
  "langgraph_checkpoint_ns",
  "thread_id",
] as const;

function _langgraphMeta(metadata: unknown): Record<string, unknown> | null {
  if (typeof metadata !== "object" || metadata === null) return null;
  const m = metadata as Record<string, unknown>;
  const extracted: Record<string, unknown> = {};
  for (const k of LANGGRAPH_META_KEYS) {
    if (k in m) extracted[k] = m[k];
  }
  return Object.keys(extracted).length > 0 ? extracted : null;
}

function _integrationFor(metadata: unknown): string {
  return _langgraphMeta(metadata) ? TraceType.LangGraph : TraceType.LangChain;
}

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

function _toJsonable(obj: unknown): unknown {
  if (obj == null) return null;
  if (typeof obj !== "object") return obj;
  if (Array.isArray(obj)) return obj.map(_toJsonable);
  // Handle LangChain message objects (they have .type and .content)
  const o = obj as Record<string, unknown>;
  if (typeof o["content"] !== "undefined" && typeof o["type"] === "string") {
    return { role: o["type"], content: o["content"] };
  }
  return Object.fromEntries(Object.entries(o).map(([k, v]) => [k, _toJsonable(v)]));
}

function _modelName(
  serialized: unknown,
  invocationParams: unknown,
  metadata: unknown
): string | null {
  const s = serialized as Record<string, unknown> | null;
  const sources: unknown[] = [
    invocationParams,
    s?.["kwargs"],
    serialized,
    metadata,
  ];
  const keys = ["model", "model_name", "deployment_name", "azure_deployment", "ls_model_name"];
  for (const source of sources) {
    if (!source || typeof source !== "object") continue;
    const o = source as Record<string, unknown>;
    for (const key of keys) {
      const val = o[key];
      if (val) return val as string;
    }
  }
  return null;
}

function _componentName(serialized: unknown): string | null {
  const s = serialized as Record<string, unknown> | null;
  if (!s) return null;
  const name = s["name"];
  if (name) return String(name);
  const id = s["id"];
  if (Array.isArray(id) && id.length > 0) return String(id[id.length - 1]);
  return null;
}

function _extractTokens(response: unknown): Record<string, unknown> | null {
  const r = response as Record<string, unknown> | null;
  if (!r) return null;
  const sources: unknown[] = [];
  const llmOutput = r["llm_output"];
  if (llmOutput && typeof llmOutput === "object") {
    const lo = llmOutput as Record<string, unknown>;
    sources.push(lo["token_usage"], lo["usage"], lo["usage_metadata"]);
  }
  // generations[0][0].message
  const msg = _firstMessage(r);
  if (msg) {
    sources.push(msg["usage_metadata"]);
    const rmd = msg["response_metadata"];
    if (rmd && typeof rmd === "object") {
      const m = rmd as Record<string, unknown>;
      sources.push(m["token_usage"], m["usage"], m["usage_metadata"]);
    }
  }
  for (const usage of sources) {
    if (!usage || typeof usage !== "object") continue;
    const u = usage as Record<string, unknown>;
    const prompt = u["prompt_tokens"] ?? u["input_tokens"];
    const completion = u["completion_tokens"] ?? u["output_tokens"];
    const total = u["total_tokens"];
    if (prompt || completion || total) {
      return { prompt: prompt ?? null, completion: completion ?? null, total: total ?? null };
    }
  }
  return null;
}

function _firstMessage(response: Record<string, unknown>): Record<string, unknown> | null {
  const gens = response["generations"] as unknown[][] | null;
  const gen = Array.isArray(gens) && Array.isArray(gens[0]) ? gens[0][0] : null;
  const msg = gen ? (gen as Record<string, unknown>)["message"] : null;
  return msg && typeof msg === "object" ? (msg as Record<string, unknown>) : null;
}

function _extractResponseText(response: unknown): string | null {
  const r = response as Record<string, unknown> | null;
  if (!r) return null;
  const gens = r["generations"] as unknown[][] | null;
  if (!Array.isArray(gens) || gens.length === 0) return null;
  // Python returns generations[0][0].text — mirror that (first generation only).
  const first = Array.isArray(gens[0]) ? (gens[0][0] as Record<string, unknown> | undefined) : undefined;
  if (!first) return null;
  const text = first["text"] ?? (first["message"] as Record<string, unknown> | null)?.["content"];
  return typeof text === "string" && text ? text : null;
}

function _extractResponseModel(response: unknown): string | null {
  const r = response as Record<string, unknown> | null;
  if (!r) return null;
  const msg = _firstMessage(r);
  const rmd = msg?.["response_metadata"];
  if (rmd && typeof rmd === "object") {
    const m = rmd as Record<string, unknown>;
    return (m["model_name"] as string) ?? (m["model"] as string) ?? null;
  }
  return null;
}

function _extractFinishReason(response: unknown): string | null {
  const r = response as Record<string, unknown> | null;
  if (!r) return null;
  const gens = r["generations"] as unknown[][] | null;
  if (!Array.isArray(gens)) return null;
  for (const row of gens) {
    if (!Array.isArray(row)) continue;
    for (const gen of row) {
      const g = gen as Record<string, unknown>;
      const info = (g["generation_info"] as Record<string, unknown> | null) ?? {};
      const reason = info["finish_reason"] ?? info["done_reason"];
      if (reason) return String(reason);
      const rmd = (g["message"] as Record<string, unknown> | null)?.["response_metadata"];
      if (rmd && typeof rmd === "object") {
        const m = rmd as Record<string, unknown>;
        const r2 = m["finish_reason"] ?? m["done_reason"];
        if (r2) return String(r2);
      }
    }
  }
  return null;
}

function _extractToolCalls(response: unknown): unknown[] | null {
  const r = response as Record<string, unknown> | null;
  if (!r) return null;
  const gens = r["generations"] as unknown[][] | null;
  if (!Array.isArray(gens)) return null;
  const calls: unknown[] = [];
  for (const row of gens) {
    if (!Array.isArray(row)) continue;
    for (const gen of row) {
      const g = gen as Record<string, unknown>;
      const msg = g["message"] as Record<string, unknown> | null;
      if (!msg) continue;
      const tcs = msg["tool_calls"];
      if (Array.isArray(tcs)) calls.push(...tcs.map(_toJsonable));
    }
  }
  return calls.length > 0 ? calls : null;
}

// ---------------------------------------------------------------------------
// Run state store (run_id → state)
// ---------------------------------------------------------------------------

interface RunState {
  start: number;
  parentId: string | null;
  model?: string | null;
  messages?: unknown;
  prompts?: unknown;
  tools?: unknown;
  metadata?: unknown;
  name?: string | null;
  input?: unknown;
}

// ---------------------------------------------------------------------------
// FluiqCallbackHandler
// ---------------------------------------------------------------------------

export class FluiqCallbackHandler {
  /** LangChain identifies handlers by name; required for it to be accepted. */
  readonly name = "fluiq_callback_handler";
  readonly ignoreLLM = false;
  readonly ignoreChain = false;
  readonly ignoreAgent = false;
  readonly ignoreRetriever = false;
  readonly ignoreCustomEvent = false;
  readonly raiseError = false;
  readonly awaitHandlers = true;

  /** LangChain calls handler.copy() when propagating to child runs. */
  copy(): FluiqCallbackHandler {
    return this;
  }

  private _runs: Map<string, RunState> = new Map();

  private _start(runId: string, state: Partial<RunState>): void {
    this._runs.set(runId, { start: Date.now() / 1000, parentId: null, ...state });
  }

  private _end(runId: string): [RunState, number] {
    const state = this._runs.get(runId) ?? { start: Date.now() / 1000, parentId: null };
    this._runs.delete(runId);
    return [state, Date.now() / 1000];
  }

  /**
   * Serializes all trace POSTs in the order they are produced.
   *
   * LangChain invokes the start/end callbacks for a run in strict order, but
   * `logTrace` POSTs are async and otherwise race: a fast node's `end` can
   * reach the backend before its `start`, which then flips the row back to
   * "running" and leaves it stuck. Chaining each emit after the previous one
   * guarantees the backend receives events (and so each node's start→end) in
   * production order. Each link swallows its own errors so one failed POST
   * never stalls the chain. Emission stays non-blocking for the user's run —
   * the callbacks still return synchronously.
   */
  private _emitChain: Promise<void> = Promise.resolve();

  private _emit(fields: Record<string, unknown>): void {
    let payload: Record<string, unknown>;
    try {
      payload = toPlainObject(fields);
    } catch {
      return; // fail open
    }
    this._emitChain = this._emitChain.then(() => logTrace(payload).catch(() => {}));
  }

  private _emitKwargs(state: RunState, fields: Record<string, unknown>): void {
    const meta = state.metadata;
    const lg = _langgraphMeta(meta);
    fields["integration"] = fields["integration"] ?? _integrationFor(meta);
    if (lg != null) fields["langgraph"] = fields["langgraph"] ?? lg;
    this._emit(fields);
  }

  private _emitStart(opts: {
    runId: string;
    parentRunId?: string | null;
    metadata?: unknown;
    type: string;
    [k: string]: unknown;
  }): void {
    const { runId, parentRunId, metadata, type: _type, ...rest } = opts;
    const lg = _langgraphMeta(metadata);
    const payload: Record<string, unknown> = {
      type: _type,
      trace_id: runId,
      parent_id: this._parent(parentRunId),
      integration: _integrationFor(metadata),
      status: "running",
      started_at: Date.now() / 1000,
      ...rest,
    };
    if (lg != null) payload["langgraph"] = lg;
    this._emit(payload);
  }

  private _parent(parentRunId?: string | null): string | null {
    if (parentRunId) return parentRunId;
    return currentParentId();
  }

  // -------------------------------------------------------------------------
  // LLM callbacks
  // -------------------------------------------------------------------------

  handleLLMStart(
    serialized: unknown,
    prompts: string[],
    runId: string,
    parentRunId?: string,
    extraParams?: unknown,
    _tags?: string[],
    metadata?: unknown,
    _runName?: string
  ): void {
    const promptJoined = prompts.filter((p) => typeof p === "string").join("\n");
    try { preCallGuard({ prompt: promptJoined }); } catch { /* fail open for tracing */ }

    enterLangchainLlm();
    const ip = (extraParams as Record<string, unknown> | null)?.["invocation_params"] as Record<string, unknown> | null;
    const model = _modelName(serialized, ip, metadata);

    this._start(runId, {
      parentId: this._parent(parentRunId),
      model,
      prompts,
      tools: ip?.["tools"] ?? null,
      metadata,
    });
    this._emitStart({
      runId,
      parentRunId,
      metadata,
      type: "llm",
      model,
      input: _toJsonable(prompts),
    });
  }

  handleChatModelStart(
    serialized: unknown,
    messages: unknown[][],
    runId: string,
    parentRunId?: string,
    extraParams?: unknown,
    _tags?: string[],
    metadata?: unknown,
    _runName?: string
  ): void {
    const flat: Record<string, unknown>[] = [];
    for (const msgList of messages) {
      const list = Array.isArray(msgList) ? msgList : [msgList];
      for (const msg of list) {
        const m = msg as Record<string, unknown>;
        let content = m["content"] ?? "";
        if (Array.isArray(content)) {
          content = content
            .map((b) =>
              typeof b === "object" && b !== null
                ? (b as Record<string, unknown>)["text"] ?? ""
                : String(b)
            )
            .join(" ");
        }
        flat.push({ role: "user", content: String(content) });
      }
    }
    try { if (flat.length) preCallGuard({ messages: flat }); } catch { /* fail open */ }

    enterLangchainLlm();
    const ip = (extraParams as Record<string, unknown> | null)?.["invocation_params"] as Record<string, unknown> | null;
    const model = _modelName(serialized, ip, metadata);

    this._start(runId, {
      parentId: this._parent(parentRunId),
      model,
      messages,
      tools: ip?.["tools"] ?? null,
      metadata,
    });
    this._emitStart({
      runId,
      parentRunId,
      metadata,
      type: "llm",
      model,
      messages: _toJsonable(messages),
    });
  }

  handleLLMEnd(response: unknown, runId: string, parentRunId?: string): void {
    const [state, end] = this._end(runId);
    exitLangchainLlm();
    this._emitKwargs(state, {
      type: "llm",
      model: state.model ?? _extractResponseModel(response),
      messages: _toJsonable(state.messages),
      input: _toJsonable(state.prompts),
      tools: _toJsonable(state.tools),
      response: _extractResponseText(response),
      tool_calls: _extractToolCalls(response),
      tokens: _extractTokens(response),
      finish_reasons: [_extractFinishReason(response)].filter(Boolean),
      latency: end - state.start,
      trace_id: runId,
      parent_id: state.parentId ?? this._parent(parentRunId),
      success: true,
    });
  }

  handleLLMError(error: Error, runId: string, parentRunId?: string): void {
    const [state, end] = this._end(runId);
    exitLangchainLlm();
    this._emitKwargs(state, {
      type: "llm",
      model: state.model,
      messages: _toJsonable(state.messages),
      input: _toJsonable(state.prompts),
      output: String(error),
      latency: end - state.start,
      trace_id: runId,
      parent_id: state.parentId ?? this._parent(parentRunId),
      success: false,
    });
  }

  // -------------------------------------------------------------------------
  // Chain callbacks
  // -------------------------------------------------------------------------

  handleChainStart(
    serialized: unknown,
    inputs: unknown,
    runId: string,
    parentRunId?: string,
    _tags?: string[],
    metadata?: unknown
  ): void {
    const name = _componentName(serialized);
    this._start(runId, {
      parentId: this._parent(parentRunId),
      name,
      input: inputs,
      metadata,
    });
    this._emitStart({
      runId,
      parentRunId,
      metadata,
      type: "chain",
      function: name,
      input: _toJsonable(inputs),
    });
  }

  handleChainEnd(outputs: unknown, runId: string, parentRunId?: string): void {
    const [state, end] = this._end(runId);
    this._emitKwargs(state, {
      type: "chain",
      function: state.name,
      input: _toJsonable(state.input),
      output: _toJsonable(outputs),
      latency: end - state.start,
      trace_id: runId,
      parent_id: state.parentId ?? this._parent(parentRunId),
      success: true,
    });
  }

  handleChainError(error: Error, runId: string, parentRunId?: string): void {
    const [state, end] = this._end(runId);
    this._emitKwargs(state, {
      type: "chain",
      function: state.name,
      input: _toJsonable(state.input),
      output: String(error),
      latency: end - state.start,
      trace_id: runId,
      parent_id: state.parentId ?? this._parent(parentRunId),
      success: false,
    });
  }

  // -------------------------------------------------------------------------
  // Tool callbacks
  // -------------------------------------------------------------------------

  handleToolStart(
    serialized: unknown,
    inputStr: string,
    runId: string,
    parentRunId?: string,
    _tags?: string[],
    metadata?: unknown
  ): void {
    const name = _componentName(serialized);
    this._start(runId, {
      parentId: this._parent(parentRunId),
      name,
      input: inputStr,
      metadata,
    });
    this._emitStart({
      runId,
      parentRunId,
      metadata,
      type: "tool",
      function: name,
      input: inputStr,
    });
  }

  handleToolEnd(output: unknown, runId: string, parentRunId?: string): void {
    const [state, end] = this._end(runId);
    this._emitKwargs(state, {
      type: "tool",
      function: state.name,
      input: state.input,
      output: _toJsonable(output),
      latency: end - state.start,
      trace_id: runId,
      parent_id: state.parentId ?? this._parent(parentRunId),
      success: true,
    });
  }

  handleToolError(error: Error, runId: string, parentRunId?: string): void {
    const [state, end] = this._end(runId);
    this._emitKwargs(state, {
      type: "tool",
      function: state.name,
      input: state.input,
      output: String(error),
      latency: end - state.start,
      trace_id: runId,
      parent_id: state.parentId ?? this._parent(parentRunId),
      success: false,
    });
  }
}

// ---------------------------------------------------------------------------
// Auto-registration — mirrors Python's register_configure_hook
// ---------------------------------------------------------------------------

let _registered = false;

/**
 * Process-wide singleton handler. `patchLangchain()` attaches it globally (see
 * `_patchCallbackManager`), so manual wiring is normally unnecessary. Still
 * exported as an escape hatch for explicit attachment
 * (`{ callbacks: [getFluiqCallbackHandler()] }`).
 */
const _handler = new FluiqCallbackHandler();

export function getFluiqCallbackHandler(): FluiqCallbackHandler {
  return _handler;
}

type CMLike = {
  configure?: (...args: unknown[]) => unknown;
  _configureSync?: (...args: unknown[]) => unknown;
  new (): unknown;
};

type ManagerLike = {
  handlers?: Array<{ name?: string }>;
  addHandler?: (h: unknown, inherit?: boolean) => void;
};

/**
 * Wrap one static factory method on CallbackManager so every manager it
 * produces carries our handler as an *inheritable* handler. Returns true if
 * the method existed and is now patched (or was already patched).
 */
function _wrapManagerFactory(CM: CMLike, methodName: "configure" | "_configureSync"): boolean {
  const fn = CM[methodName] as
    | (((...args: unknown[]) => unknown) & { _fluiqPatched?: boolean })
    | undefined;
  if (typeof fn !== "function") return false;
  if (fn._fluiqPatched) return true;

  const original = fn.bind(CM);

  const patched = function (this: unknown, ...args: unknown[]): unknown {
    let manager = original(...args) as ManagerLike | undefined;
    try {
      // The factory returns undefined when there are no handlers and tracing
      // is disabled — create a manager so our handler still runs.
      if (!manager) manager = new (CM as { new (): ManagerLike })();
      const handlers = manager.handlers;
      const already =
        Array.isArray(handlers) && handlers.some((h) => h && h.name === _handler.name);
      if (!already && typeof manager.addHandler === "function") {
        // inherit = true so child runs (LangGraph nodes, sub-chains) keep it.
        manager.addHandler(_handler, true);
      }
    } catch {
      // fail open — never break the user's LangChain run
    }
    return manager;
  } as ((...args: unknown[]) => unknown) & { _fluiqPatched?: boolean };

  patched._fluiqPatched = true;
  (CM as unknown as Record<string, unknown>)[methodName] = patched;
  return true;
}

/**
 * Inject the singleton handler into LangChain's central callback factory.
 *
 * Python attaches the handler globally via `register_configure_hook`, which
 * does not exist in `@langchain/core` (JS). The JS chokepoint is the static
 * `CallbackManager._configureSync(...)` — `getCallbackManagerForConfig()` (used
 * by every Runnable, and therefore every LangGraph node / tool / LLM call)
 * calls it directly, and the public `configure()` delegates to it. We wrap it
 * so our handler is added as an inheritable handler on every manager, mirroring
 * Python's inheritable hook. `configure` is also wrapped for older cores that
 * predate `_configureSync`. Returns true if at least one method was patched.
 */
function _patchCallbackManager(): boolean {
  const mod = require("@langchain/core/callbacks/manager") as {
    CallbackManager?: CMLike;
  };
  const CM = mod.CallbackManager;
  if (!CM) return false;

  // _configureSync is the real chokepoint in current cores; configure is a
  // fallback for versions that only expose the public method.
  const a = _wrapManagerFactory(CM, "_configureSync");
  const b = _wrapManagerFactory(CM, "configure");
  return a || b;
}

export function patchLangchain(): void {
  if (_registered) return;

  try {
    const ok = _patchCallbackManager();
    // Only mark as registered once the patch lands. If @langchain/core is not
    // installed yet (e.g. instrument() called before the dep loads) we leave
    // _registered false so a later call can retry.
    if (ok) _registered = true;
  } catch {
    // @langchain/core not installed — silently skip, allow retry later
  }
}

/** LangGraph executes through LangChain Core's callbacks; same handler, idempotent. */
export function patchLangGraph(): void {
  patchLangchain();
}
