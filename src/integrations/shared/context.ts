import { AsyncLocalStorage } from "async_hooks";

/**
 * Mutable context stored per async execution chain.
 * AsyncLocalStorage propagates the same object reference to all async
 * children, so mutations to the object are visible within the same chain.
 */
export interface TraceCtx {
  parentId: string | null;
  llmTraceId: string | null;
}

const _storage = new AsyncLocalStorage<TraceCtx>();

function _getCtx(): TraceCtx {
  return _storage.getStore() ?? { parentId: null, llmTraceId: null };
}

export function currentParentId(): string | null {
  return _getCtx().parentId;
}

export function currentLlmTraceId(): string | null {
  return _getCtx().llmTraceId;
}

/**
 * Ambient "inside a LangChain LLM call" depth.
 *
 * When LangChain/LangGraph invokes a chat model it ultimately calls the raw
 * provider SDK (OpenAI/Anthropic/Gemini), which we also patch — so the same
 * call would be traced twice. The LangChain callback handler brackets each LLM
 * call with enter/exit, and the raw SDK patches skip emitting while the depth
 * is > 0, leaving the richer LangChain trace as the single source.
 *
 * Python uses a ContextVar (ambient, always settable). JS AsyncLocalStorage
 * cannot set a value without an active `.run()` scope — and LangGraph runs are
 * not wrapped in one — so a context-stored flag never took effect. A
 * module-level counter is the ambient equivalent. Node is single-threaded, so
 * increments/decrements are atomic; a counter (not a boolean) keeps nested and
 * concurrent LangChain LLM calls balanced.
 */
let _langchainLlmDepth = 0;

/** Returns true when a LangChain callback is handling an LLM call (prevents double-tracing). */
export function isInLangchainLlm(): boolean {
  return _langchainLlmDepth > 0;
}

/** Mark entry into a LangChain LLM callback. */
export function enterLangchainLlm(): void {
  _langchainLlmDepth++;
}

/** Mark exit from a LangChain LLM callback. */
export function exitLangchainLlm(): void {
  if (_langchainLlmDepth > 0) _langchainLlmDepth--;
}

/**
 * Run `fn` in a child context where `traceId` is the new parentId.
 * Returns both the result and the child context.
 */
export function runInChildContext<T>(
  traceId: string,
  fn: () => Promise<T>
): { promise: Promise<{ result: T; ctx: TraceCtx }>; ctx: TraceCtx } {
  const parent = _getCtx();
  const childCtx: TraceCtx = {
    parentId: traceId,
    llmTraceId: parent.llmTraceId,
  };
  const promise = _storage.run(childCtx, async () => {
    const result = await fn();
    return { result, ctx: childCtx };
  });
  return { promise, ctx: childCtx };
}

/**
 * Set `llmTraceId` in the current context for the full duration of `fn`,
 * then restore the previous value. Used by integration patches to mark
 * the in-flight LLM trace ID so the completion trace (and nested calls,
 * e.g. MCP) reuse the same ID as the "running" trace.
 *
 * `fn` is awaited before the previous value is restored. This matters when
 * `fn` is async: without awaiting, the `finally` would restore `llmTraceId`
 * at `fn`'s first `await`, so a completion `logTrace()` running later would
 * miss the context ID and mint a fresh random `trace_id` — emitting a second
 * trace instead of updating the running one.
 */
export async function withLlmTraceId<T>(traceId: string, fn: () => T | Promise<T>): Promise<T> {
  const ctx = _storage.getStore();
  if (!ctx) return fn();
  const prev = ctx.llmTraceId;
  ctx.llmTraceId = traceId;
  try {
    return await fn();
  } finally {
    ctx.llmTraceId = prev;
  }
}

/**
 * Run `fn` inside a fresh root context (used by integration patches that
 * need a context when none has been established by a @trace wrapper).
 * Returns the context object so callers can read mutations after fn returns.
 */
export function runInRootContext<T>(fn: () => Promise<T>): Promise<T> {
  const existing = _storage.getStore();
  if (existing) return fn();
  const rootCtx: TraceCtx = { parentId: null, llmTraceId: null };
  return _storage.run(rootCtx, fn);
}
