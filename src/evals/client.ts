/**
 * Thin SDK client for block-mode evaluation.
 *
 * callEvaluateBlock() — synchronously calls /evaluate and returns
 * {metric: score} or null. Only used in block mode; warn-mode evaluation
 * is handled entirely server-side via the Kafka → worker pipeline.
 *
 * Never raises — failures are logged and swallowed. The caller in tracer.ts
 * inspects the returned scores and raises FluiqEvalError when appropriate.
 */
import axios from "axios";
import { _config } from "../config";

function _extractQuestion(trace: Record<string, unknown>): string {
  const messages =
    trace["messages"] ?? trace["contents"] ?? trace["input"] ?? [];

  if (typeof messages === "string") return messages;

  if (Array.isArray(messages)) {
    // Prefer last user message
    for (let i = messages.length - 1; i >= 0; i--) {
      const msg = messages[i];
      if (typeof msg === "object" && msg !== null) {
        const m = msg as Record<string, unknown>;
        if (m["role"] === "user") {
          const content = m["content"];
          if (typeof content === "string" && content.trim()) return content;
        }
      }
    }
    // Fall back to concatenating all content
    const parts = messages
      .filter((m): m is Record<string, unknown> => typeof m === "object" && m !== null)
      .map((m) => String(m["content"] ?? ""))
      .filter(Boolean);
    return parts.join("\n");
  }

  return messages ? String(messages) : "";
}

export async function callEvaluateBlock(
  trace: Record<string, unknown>
): Promise<Record<string, number> | null> {
  try {
    const metrics = _config.eval_metrics ?? ["hallucination", "relevance"];
    const judgeModel = _config.eval_judge_model;
    const thresholds = _config.eval_thresholds;

    const base = `${_config.endpoint}/${_config.version}`;
    const response = await axios.post(
      `${base}/evaluate`,
      {
        api_key: _config.api_key,
        trace_id: trace["trace_id"] ?? null,
        model: trace["model"] ?? "",
        prompt: _extractQuestion(trace),
        response: trace["response"] ?? trace["output"] ?? "",
        context: "",
        metrics: [...metrics],
        judge_model: judgeModel,
        thresholds: { ...thresholds },
      },
      { timeout: 30_000, validateStatus: () => true }
    );

    if (response.status === 200) {
      const data = response.data as Record<string, unknown>;
      return (data["scores"] as Record<string, number>) ?? {};
    }

    // Non-200 from /evaluate — fail open (no block).
    return null;
  } catch {
    // Block-mode evaluation failed — fail open (no block).
    return null;
  }
}
