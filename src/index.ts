/**
 * Fluiq TypeScript SDK
 *
 * Instrument any LLM application in two lines. Auto-tracing for OpenAI and
 * Anthropic — plus a `trace()` wrapper for everything else. Every run is
 * cost-tracked in your dashboard. Add one more line to enable security
 * scanning, evaluation, or Redis caching.
 *
 * @example
 * import fluiq from "fluiq";
 * fluiq.instrument({ apiKey: "fl_..." });
 */

import axios from "axios";
import { API_KEY, ENDPOINT, VERSION, _config, init, authHeaders } from "./config";
import { FluiqSecurityError, FluiqEvalError } from "./exceptions";
import { flushEvents } from "./client";
import { trace } from "./trace";
import { Prompt } from "./prompts";

export { FluiqSecurityError, FluiqEvalError } from "./exceptions";
export { trace } from "./trace";
export { Prompt } from "./prompts";

// ---------------------------------------------------------------------------
// instrument()
// ---------------------------------------------------------------------------

export interface InstrumentOptions {
  /** Your Fluiq API key. Defaults to the FLUIQ_API_KEY env var. */
  apiKey?: string;
  /** Override the ingest endpoint. Defaults to FLUIQ_API_ENDPOINT or https://api.getfluiq.com/api. */
  endpoint?: string;
  /** Trace schema version. Pin in production to opt in to schema bumps explicitly. */
  version?: string;
}

/**
 * Start Fluiq instrumentation.
 *
 * Must be called before any LLM API calls. Patches the OpenAI and Anthropic
 * SDKs if they are installed. Reads FLUIQ_API_KEY from the environment if
 * apiKey is not provided.
 */
let _drainRegistered = false;

export function instrument(options: InstrumentOptions = {}): void {
  init({
    api_key: options.apiKey ?? API_KEY,
    endpoint: options.endpoint ?? ENDPOINT,
    version: options.version ?? VERSION,
  });

  // Drain any fire-and-forget trace POSTs that are still in flight when the
  // process is about to exit, so short-lived scripts (e.g. a one-shot
  // vectorstore search whose POST is kicked off right before the program ends)
  // don't lose their final traces. `beforeExit` does not fire on explicit
  // process.exit(); call `flushEvents()` directly in that case.
  if (!_drainRegistered && typeof process !== "undefined" && typeof process.once === "function") {
    _drainRegistered = true;
    process.once("beforeExit", () => {
      void flushEvents();
    });
  }
}

// ---------------------------------------------------------------------------
// optimize()
// ---------------------------------------------------------------------------

export interface OptimizeOptions {
  /**
   * "cache"   (default) — full caching enabled. Repeated LLM calls that match
   *                        the backend profile are intercepted and served from Redis.
   * "observe"            — no interception. Records what would have been a hit.
   */
  mode?: "cache" | "observe";
}

/**
 * Activate trace-driven Redis caching (requires Team plan or above).
 * Must be called after instrument().
 *
 * Fluiq's backend analyses your historical traces to determine which LLM
 * calls are repeated most often and provisions a dedicated Redis instance.
 * On the first call after optimize() the SDK fetches the cache profile and
 * begins serving repeated prompts from cache — saving both latency and cost.
 */
export function optimize(options: OptimizeOptions = {}): void {
  const mode = options.mode ?? "cache";
  if (mode !== "cache" && mode !== "observe") {
    throw new Error(`fluiq.optimize() mode must be 'cache' or 'observe', got '${mode}'`);
  }
  _config.optimize = true;
  _config.optimize_mode = mode;
}

// ---------------------------------------------------------------------------
// eval()
// ---------------------------------------------------------------------------

export interface EvalOptions {
  /**
   * Per-metric pass/fail thresholds, e.g. { hallucination: 0.8, relevance: 0.7 }.
   * Supported: hallucination, faithfulness, relevance, toxicity, coherence, completeness.
   */
  thresholds?: Record<string, number>;
  /**
   * Which metrics to evaluate. Defaults to ["hallucination", "relevance"] when omitted.
   */
  metrics?: string[];
  /**
   * "warn"  (default) — evaluate in the background and log a warning on threshold failure.
   * "block"           — evaluate synchronously after each LLM call and raise FluiqEvalError on failure.
   */
  mode?: "warn" | "block";
  /** The model Fluiq uses as judge. Defaults to "claude-haiku-4-5-20251001". */
  judgeModel?: string;
}

/**
 * Activate server-side LLM response evaluation.
 * Must be called after instrument().
 *
 * After each LLM call Fluiq runs an LLM-as-judge on the response, scores
 * each requested metric (0 = worst, 1 = best), stores the results in your
 * dashboard, and — depending on mode — either warns or blocks when a score
 * falls below its threshold.
 */
export function fluiqEval(options: EvalOptions = {}): void {
  const mode = options.mode ?? "warn";
  if (mode !== "warn" && mode !== "block") {
    throw new Error(`fluiq.eval() mode must be 'warn' or 'block', got '${mode}'`);
  }
  _config.eval = true;
  _config.eval_mode = mode;
  _config.eval_thresholds = options.thresholds ? { ...options.thresholds } : {};
  _config.eval_metrics = options.metrics ? [...options.metrics] : ["hallucination", "relevance"];
  _config.eval_judge_model = options.judgeModel ?? "claude-haiku-4-5-20251001";
}

// ---------------------------------------------------------------------------
// secure()
// ---------------------------------------------------------------------------

export interface SecureOptions {
  /**
   * "warn"  (default) — post-call scan only. Security fields are written into
   *                     the stored trace. Your LLM calls are never interrupted.
   * "block"           — pre-call guard enabled. Every prompt is checked before
   *                     the LLM API call. FluiqSecurityError is raised on detection.
   */
  mode?: "warn" | "block";
  /**
   * Slug of the named guardrail policy to use (configured in the dashboard).
   * Defaults to "default". Unknown slugs fall back to "default" server-side.
   */
  guardrail?: string;
}

/**
 * Activate server-side security scanning (requires Team plan or above).
 * Must be called after instrument().
 *
 * Scans for PII, prompt injection, jailbreaks, skeleton-key attacks, leaked
 * secrets, and indirect injection in tool outputs. Detection runs on the
 * Fluiq backend — patterns are never shipped in the public SDK.
 *
 * Free-tier keys receive a 402 and fall back to warn behaviour automatically.
 */
export function secure(options: SecureOptions = {}): void {
  const mode = options.mode ?? "warn";
  if (mode !== "warn" && mode !== "block") {
    throw new Error(`fluiq.secure() mode must be 'warn' or 'block', got '${mode}'`);
  }
  _config.secure = true;
  _config.secure_mode = mode;
  _config.secure_guardrail = options.guardrail ?? "default";
}

// ---------------------------------------------------------------------------
// fetchPrompt()
// ---------------------------------------------------------------------------

export interface FetchPromptOptions {
  /** Which environment snapshot to load. Defaults to "production". */
  env?: "production" | "staging" | "development";
}

/**
 * Fetch a deployed prompt template from the Fluiq dashboard.
 * Must be called after instrument().
 *
 * @param slug The prompt's URL-safe identifier as set in the dashboard.
 * @returns A Prompt whose `.render(variables)` substitutes `{variable}` placeholders.
 */
export async function fetchPrompt(
  slug: string,
  options: FetchPromptOptions = {}
): Promise<Prompt> {
  const env = options.env ?? "production";
  const url = `${_config.endpoint}/${_config.version}/prompts/fetch/${slug}`;
  const resp = await axios.get(url, {
    params: { env },
    headers: authHeaders(),
    timeout: 10000,
  });
  return new Prompt(resp.data);
}

// ---------------------------------------------------------------------------
// lookupToolResult()
// ---------------------------------------------------------------------------

/**
 * Return a cached tool result, or `null` if not in cache.
 *
 * `args` can be an object or a JSON string. Keys are sorted before hashing so
 * argument order does not matter.
 *
 * @example
 * let result = await fluiq.lookupToolResult("get_weather", { location: "London" });
 * if (result == null) result = await callWeatherApi("London");
 */
export async function lookupToolResult(
  toolName: string,
  args: Record<string, unknown> | string
): Promise<unknown> {
  try {
    const { lookupToolCache } = require("./optimization/client") as typeof import("./optimization/client");
    return await lookupToolCache(toolName, args);
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Default export (fluiq namespace object)
// ---------------------------------------------------------------------------

const fluiq = {
  instrument,
  optimize,
  eval: fluiqEval,
  secure,
  trace,
  fetchPrompt,
  lookupToolResult,
  Prompt,
  FluiqSecurityError,
  FluiqEvalError,
};

export default fluiq;
