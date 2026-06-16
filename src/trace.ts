/**
 * @fluiq.trace equivalent for TypeScript.
 *
 * Wraps any sync or async function to record its inputs, outputs, latency,
 * and errors. Supports nested calls — the parent trace ID is automatically
 * propagated through AsyncLocalStorage so child LLM calls can reference it.
 *
 * Usage:
 *   const myFn = fluiq.trace(async (x: string) => { ... });
 *   // or with a custom name:
 *   const myFn = fluiq.trace(async (x: string) => { ... }, { name: "research_agent" });
 */
import { randomUUID } from "crypto";
import { logTrace } from "./tracer";
import { _config } from "./config";
import { TraceType, toPlainObject } from "./integrations/shared/models";
import { currentParentId, runInChildContext } from "./integrations/shared/context";

// ---------------------------------------------------------------------------
// Core wrapper builder
// ---------------------------------------------------------------------------

function _buildWrapper<T extends (...args: unknown[]) => unknown>(
  fn: T,
  funcName: string
): T {
  const isAsync =
    fn.constructor.name === "AsyncFunction" ||
    (fn.toString().startsWith("async ") ?? false);

  if (isAsync) {
    const asyncWrapper = async function (
      this: unknown,
      ...args: unknown[]
    ): Promise<unknown> {
      const traceId = randomUUID();
      const parentId = currentParentId();
      const start = Date.now() / 1000;
      const argsKey = _serializeArgs(args);

      const emitStart = () =>
        logTrace(
          toPlainObject({
            trace_id: traceId,
            parent_id: parentId,
            integration: TraceType.GeneralFunction,
            function: funcName,
            type: "function",
            input: argsKey,
            status: "running",
            started_at: start,
          })
        ).catch(() => {});

      // Optimize path: serve a previously cached return value without running
      // the function. lookupFunctionCache returns a wrapper object so a cached
      // `null`/`undefined` result is distinguishable from a miss.
      if (_config.optimize) {
        let cached: Record<string, unknown> | null = null;
        try {
          const { lookupFunctionCache } = require("./optimization/client") as typeof import("./optimization/client");
          cached = await lookupFunctionCache(funcName, argsKey);
        } catch {
          cached = null;
        }
        if (cached != null) {
          const cachedResult = cached["result"];
          const end = Date.now() / 1000;
          await emitStart();
          await logTrace(
            toPlainObject({
              trace_id: traceId,
              parent_id: parentId,
              integration: TraceType.GeneralFunction,
              function: funcName,
              type: "function",
              input: argsKey,
              output: _serializeResult(cachedResult),
              latency: end - start,
              success: true,
              status: "success",
              started_at: start,
              _cache_hit: true,
            })
          ).catch(() => {});
          return cachedResult;
        }
      }

      await emitStart();

      let innerResult: unknown;
      let exc: Error | null = null;
      let cacheHit = false;

      const { promise } = runInChildContext(traceId, async () => {
        innerResult = await (fn as (...a: unknown[]) => Promise<unknown>).apply(this, args);
        return innerResult;
      });

      try {
        const { result, ctx } = await promise;
        innerResult = result;
        cacheHit = ctx.innerCacheHit;
      } catch (e) {
        exc = e as Error;
      }

      const end = Date.now() / 1000;
      const success = exc === null;

      if (_config.optimize && success) {
        try {
          const { populateFunctionCache } = require("./optimization/client") as typeof import("./optimization/client");
          await populateFunctionCache(funcName, argsKey, innerResult);
        } catch {
          // caching must never crash the traced function
        }
      }

      await logTrace(
        toPlainObject({
          trace_id: traceId,
          parent_id: parentId,
          integration: TraceType.GeneralFunction,
          function: funcName,
          type: "function",
          input: argsKey,
          output: success ? _serializeResult(innerResult) : String(exc),
          latency: end - start,
          success,
          status: success ? "success" : "error",
          started_at: start,
          ...(cacheHit ? { _cache_hit: true } : {}),
        })
      ).catch(() => {});

      if (exc) throw exc;
      return innerResult;
    };

    Object.defineProperty(asyncWrapper, "name", { value: fn.name || funcName });
    return asyncWrapper as unknown as T;
  }

  // Synchronous wrapper
  const syncWrapper = function (this: unknown, ...args: unknown[]): unknown {
    const traceId = randomUUID();
    const parentId = currentParentId();
    const start = Date.now() / 1000;

    logTrace(
      toPlainObject({
        trace_id: traceId,
        parent_id: parentId,
        integration: TraceType.GeneralFunction,
        function: funcName,
        type: "function",
        input: _serializeArgs(args),
        status: "running",
        started_at: start,
      })
    ).catch(() => {});

    let result: unknown;
    let exc: Error | null = null;

    try {
      result = (fn as (...a: unknown[]) => unknown).apply(this, args);
    } catch (e) {
      exc = e as Error;
    }

    const end = Date.now() / 1000;
    const success = exc === null;

    logTrace(
      toPlainObject({
        trace_id: traceId,
        parent_id: parentId,
        integration: TraceType.GeneralFunction,
        function: funcName,
        type: "function",
        input: _serializeArgs(args),
        output: success ? _serializeResult(result) : String(exc),
        latency: end - start,
        success,
        status: success ? "success" : "error",
        started_at: start,
      })
    ).catch(() => {});

    if (exc) throw exc;
    return result;
  };

  Object.defineProperty(syncWrapper, "name", { value: fn.name || funcName });
  return syncWrapper as unknown as T;
}

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

function _serializeArgs(args: unknown[]): string {
  try {
    return JSON.stringify(args);
  } catch {
    return String(args);
  }
}

function _serializeResult(result: unknown): string {
  if (result === undefined) return "undefined";
  try {
    return JSON.stringify(result);
  } catch {
    return String(result);
  }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export interface TraceOptions {
  name?: string;
}

/**
 * Wrap a function with Fluiq tracing. Works with both sync and async functions.
 *
 * @example
 * const tracedFn = trace(async (query: string) => callLLM(query));
 * const named = trace(async (q: string) => callLLM(q), { name: "my_agent" });
 */
export function trace<T extends (...args: unknown[]) => unknown>(
  fn: T,
  options?: TraceOptions
): T {
  const funcName = options?.name ?? fn.name ?? "anonymous";
  return _buildWrapper(fn, funcName);
}
