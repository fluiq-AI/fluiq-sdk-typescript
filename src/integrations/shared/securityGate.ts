import { _config } from "../../config";

function _extractPrompt(params: Record<string, unknown>): string {
  // OpenAI chat completions
  const messages = params["messages"];
  if (Array.isArray(messages) && messages.length > 0) {
    const parts: string[] = [];
    for (const m of messages) {
      if (typeof m !== "object" || m === null) continue;
      const msg = m as Record<string, unknown>;
      const content = msg["content"];
      if (typeof content === "string") {
        parts.push(content);
      } else if (Array.isArray(content)) {
        for (const block of content) {
          if (typeof block === "object" && block !== null) {
            const b = block as Record<string, unknown>;
            if (b["type"] === "text" && typeof b["text"] === "string") {
              parts.push(b["text"]);
            }
          }
        }
      }
    }
    return parts.join("\n");
  }

  // OpenAI responses API / Gemini
  const inp = params["input"] ?? params["contents"];
  if (typeof inp === "string") return inp;
  if (Array.isArray(inp)) return inp.map(String).join("\n");

  // Anthropic
  const prompt = params["prompt"];
  if (typeof prompt === "string") return prompt;

  return "";
}

export async function preCallGuard(params: Record<string, unknown>): Promise<void> {
  if (!_config.secure) return;
  if (_config.secure_mode !== "block") return;
  if (!_config.api_key) return;

  const prompt = _extractPrompt(params);
  if (!prompt.trim()) return;

  const { preCallCheck } = require("../../security/client") as typeof import("../../security/client");
  await preCallCheck(prompt);
}
