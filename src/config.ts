export const API_KEY: string | null = process.env.FLUIQ_API_KEY ?? null;
export const ENDPOINT: string = process.env.FLUIQ_API_ENDPOINT ?? "https://api.getfluiq.com/api";
export const VERSION = "v1";

export interface FluiqConfig {
  api_key: string | null;
  enabled: boolean;
  version: string;
  endpoint: string;
  secure: boolean;
  secure_mode: "warn" | "block";
  secure_guardrail: string;
  optimize: boolean;
  optimize_mode: "cache" | "observe";
  eval: boolean;
  eval_mode: "warn" | "block";
  eval_metrics: string[] | null;
  eval_thresholds: Record<string, number>;
  eval_judge_model: string;
  eval_custom_judges: Record<string, number>;
}

export const _config: FluiqConfig = {
  api_key: null,
  enabled: true,
  version: VERSION,
  endpoint: ENDPOINT,
  secure: false,
  secure_mode: "warn",
  secure_guardrail: "default",
  optimize: false,
  optimize_mode: "cache",
  eval: false,
  eval_mode: "warn",
  eval_metrics: null,
  eval_thresholds: {},
  eval_judge_model: "claude-haiku-4-5-20251001",
  eval_custom_judges: {},
};

/**
 * Return the Authorization header carrying the configured API key.
 *
 * The API key is transmitted as an HTTP `Authorization: Bearer` token on every
 * SDK → fluiq-api request. Returns an empty object when no key is configured so
 * callers can spread it unconditionally.
 */
export function authHeaders(): Record<string, string> {
  const key = _config.api_key;
  return key ? { Authorization: `Bearer ${key}` } : {};
}

export function init(options: {
  api_key?: string | null;
  version?: string;
  endpoint?: string;
} = {}): void {
  if (options.api_key !== undefined) _config.api_key = options.api_key ?? null;
  if (options.version !== undefined) _config.version = options.version;
  if (options.endpoint !== undefined) _config.endpoint = options.endpoint;

  // Lazy import to avoid circular deps
  const { initIntegrations } = require("./integrations") as typeof import("./integrations");
  initIntegrations();
}
