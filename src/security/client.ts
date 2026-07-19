/**
 * Thin SDK client for the /secure/check endpoint.
 *
 * preCallCheck() — pre-call synchronous guard: sends the prompt to the
 * server before the LLM call. Raises FluiqSecurityError when mode='block'
 * and the server returns allow=false. In all other cases (network error,
 * 402, warn mode) returns silently so the LLM call proceeds normally.
 *
 * Post-call security scanning is handled asynchronously by the evaluator
 * worker. The SDK embeds _security_config in the trace event; /ingest strips
 * it and fans out an sdk_security job to the evaluator Kafka topic.
 */
import axios from "axios";
import { _config, authHeaders } from "../config";
import { FluiqSecurityError } from "../exceptions";
import { currentLlmTraceId } from "../integrations/shared/context";

function _baseUrl(): string {
  return `${_config.endpoint}/${_config.version}`;
}

export async function preCallCheck(promptText: string): Promise<void> {
  const mode = _config.secure_mode;
  try {
    // Authenticate via the Bearer header (like every other SDK request) rather
    // than only putting the key in the body — bodies are far more likely to be
    // captured by proxy/APM logs. Send trace_id + guardrail for Python parity so
    // the server can honor a custom guardrail and publish the blocked trace.
    const response = await axios.post(
      `${_baseUrl()}/secure/check`,
      {
        prompt: promptText,
        trace_id: currentLlmTraceId(),
        guardrail: _config.secure_guardrail,
      },
      { timeout: 2000, validateStatus: () => true, headers: authHeaders() }
    );

    if (response.status === 402) {
      // Plan doesn't include secure — silently fall back to warn behaviour.
      return;
    }

    const result = response.data as Record<string, unknown>;
    if (result["allow"] === false && mode === "block") {
      throw new FluiqSecurityError(
        String(result["block_reason"] ?? "Blocked by fluiq.secure"),
        String(result["risk_level"] ?? "high"),
        Array.isArray(result["attack_types"]) ? (result["attack_types"] as string[]) : []
      );
    }
  } catch (exc) {
    if (exc instanceof FluiqSecurityError) throw exc;
    // Network / infra failures are swallowed — fail open.
  }
}
