import { randomUUID } from "crypto";
import { sendEvent, sendEventGated } from "./client";
import { _config } from "./config";
import { currentLlmTraceId, currentParentId } from "./integrations/shared/context";
import { computeChainId } from "./integrations/shared/chain";
import { FluiqEvalError, FluiqSecurityError } from "./exceptions";

export async function logTrace(data: Record<string, unknown>): Promise<void> {
  try {
    data["timestamp"] = Date.now() / 1000;

    if (!data["trace_id"]) {
      const ctxTraceId = currentLlmTraceId();
      data["trace_id"] = ctxTraceId ?? randomUUID();
    }

    if (!data["parent_id"]) {
      const ctxParent = currentParentId();
      if (ctxParent) {
        data["parent_id"] = ctxParent;
      } else {
        const chainId = computeChainId(data);
        if (chainId != null) {
          data["parent_id"] = chainId;
        }
      }
    }

    // Remove legacy local-scan flag if present (no-op, kept for compat)
    delete data["_security_scan"];

    const isPreBlocked = Boolean(data["_security_pre_blocked"]);
    delete data["_security_pre_blocked"];

    if (_config.secure && !isPreBlocked) {
      // Embed security config so /ingest fans out to the evaluator worker.
      // The worker runs the full post-call scan asynchronously.
      data["_security_config"] = {
        mode: _config.secure_mode,
        guardrail: _config.secure_guardrail,
      };
    }

    const isCacheHit = data["_cache_hit"] as boolean | undefined;
    delete data["_cache_hit"];

    if (isCacheHit) {
      data["cache_hit"] = true;
    } else if (
      _config.optimize &&
      (data["type"] === "llm" || data["type"] === "function") &&
      data["latency"] != null
    ) {
      data["cache_hit"] = false;
    }

    if (_config.optimize && !isCacheHit) {
      try {
        const { populateCache } = require("./optimization/client") as typeof import("./optimization/client");
        populateCache(data);
      } catch (_) {
        // ignore
      }
    }

    const responseStr = _extractResponseStr(data);

    // Warn mode: embed eval config so /ingest fans out to the eval worker
    if (_config.eval && !isCacheHit && data["type"] === "llm" && responseStr.trim()) {
      if (_config.eval_mode === "warn") {
        data["_eval_config"] = {
          metrics: _config.eval_metrics ?? ["hallucination", "relevance"],
          judge_model: _config.eval_judge_model,
          thresholds: _config.eval_thresholds,
        };
      }
    }

    // Response gate: when secure mode='block', read /ingest's return value so
    // we can raise FluiqSecurityError before the LLM output reaches the caller.
    // Warm path (scan_responses=false on the server) returns {} immediately.
    const useGate =
      _config.secure &&
      _config.secure_mode === "block" &&
      !isPreBlocked &&
      data["type"] === "llm";
    if (useGate) {
      const gate = await sendEventGated(data);
      if (gate["response_blocked"]) {
        throw new FluiqSecurityError(
          (gate["block_reason"] as string) ?? "Response blocked by fluiq.secure()",
          (gate["risk_level"] as string) ?? "high",
          (gate["attack_types"] as string[]) ?? []
        );
      }
    } else {
      await sendEvent(data);
    }

    // Block mode: synchronous /evaluate call after trace is stored
    if (_config.eval && !isCacheHit && _config.eval_mode === "block") {
      if (data["type"] === "llm" && responseStr.trim()) {
        const { callEvaluateBlock } = require("./evals/client") as typeof import("./evals/client");
        const scores = await callEvaluateBlock(data);
        if (scores) {
          const thresholds = _config.eval_thresholds;
          const failures: Record<string, number> = {};
          for (const [metric, score] of Object.entries(scores)) {
            if (score < (thresholds[metric] ?? 0.0)) {
              failures[metric] = score;
            }
          }
          if (Object.keys(failures).length > 0) {
            throw new FluiqEvalError(failures, scores);
          }
        }
      }
    }
  } catch (exc) {
    if (exc instanceof FluiqEvalError || exc instanceof FluiqSecurityError) throw exc;
    // Swallow all other errors — SDK must never crash the application
  }
}

function _extractResponseStr(data: Record<string, unknown>): string {
  const resp = data["response"];
  if (typeof resp === "string") return resp;
  if (Array.isArray(resp)) return resp.map((x) => String(x)).join(" ");
  return "";
}
