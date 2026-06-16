import { _config } from "../../config";
import { markInnerCacheHit } from "./context";

function _isCacheActive(): boolean {
  return (
    _config.optimize &&
    _config.optimize_mode === "cache" &&
    !!_config.api_key
  );
}

export async function preCallOptimize(
  params: Record<string, unknown>,
  provider: "openai" | "anthropic" | "gemini"
): Promise<Record<string, unknown> | null> {
  if (!_isCacheActive()) return null;

  try {
    const { lookupCache } = require("../../optimization/client") as typeof import("../../optimization/client");
    const payload = await lookupCache(params);
    if (payload == null) return null;

    markInnerCacheHit();
    return payload;
  } catch {
    return null;
  }
}

export async function preCallOptimizeEmbedding(
  params: Record<string, unknown>
): Promise<Record<string, unknown> | null> {
  if (!_isCacheActive()) return null;

  try {
    const { lookupEmbeddingCache } = require("../../optimization/client") as typeof import("../../optimization/client");
    const payload = await lookupEmbeddingCache(params);
    if (payload == null) return null;

    markInnerCacheHit();
    return payload;
  } catch {
    return null;
  }
}
