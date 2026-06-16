/**
 * Automatic tool-result learning from LLM conversation history.
 *
 * Called pre-call in each provider patch. Scans the incoming messages/contents
 * for (tool_call → tool_result) pairs and populates the tool cache so future
 * identical tool calls can be served via `fluiq.lookupToolResult(name, args)`.
 *
 * Mirrors fluiq.integrations.shared.tool_cache from the Python SDK.
 */

function _tryPopulate(toolName: string, args: unknown, result: unknown): void {
  try {
    const { populateToolCache } = require("../../optimization/client") as typeof import("../../optimization/client");
    // Fire-and-forget: caching must never block or crash the LLM call.
    void populateToolCache(toolName, args as Record<string, unknown> | string, result).catch(
      () => {}
    );
  } catch {
    // optimization client unavailable — skip silently
  }
}

function _asRecord(v: unknown): Record<string, unknown> | null {
  return v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
}

/** Scan OpenAI-format messages for tool_call/tool_result pairs and cache. */
export function learnFromOpenAIMessages(messages: unknown): void {
  if (!Array.isArray(messages) || messages.length === 0) return;
  const pending = new Map<string, [string, string]>();

  for (const raw of messages) {
    const msg = _asRecord(raw);
    if (!msg || msg["role"] !== "assistant") continue;
    const toolCalls = msg["tool_calls"];
    if (!Array.isArray(toolCalls)) continue;
    for (const tcRaw of toolCalls) {
      const tc = _asRecord(tcRaw) ?? (tcRaw as Record<string, unknown>);
      if (!tc) continue;
      const tcId = (tc["id"] as string) ?? (tc as { id?: string }).id;
      const fn = _asRecord(tc["function"]) ?? (tc as { function?: Record<string, unknown> }).function;
      const name = fn ? (fn["name"] as string) : undefined;
      const args = fn ? ((fn["arguments"] as string) ?? "{}") : "{}";
      if (tcId && name) pending.set(tcId, [name, args]);
    }
  }

  for (const raw of messages) {
    const msg = _asRecord(raw);
    if (!msg || msg["role"] !== "tool") continue;
    const tcId = msg["tool_call_id"] as string | undefined;
    const content = msg["content"] ?? "";
    if (tcId && pending.has(tcId)) {
      const [name, args] = pending.get(tcId)!;
      _tryPopulate(name, args, content);
    }
  }
}

/** Scan Anthropic-format messages for tool_use/tool_result pairs and cache. */
export function learnFromAnthropicMessages(messages: unknown): void {
  if (!Array.isArray(messages) || messages.length === 0) return;
  const pending = new Map<string, [string, unknown]>();

  for (const raw of messages) {
    const msg = _asRecord(raw);
    if (!msg || msg["role"] !== "assistant") continue;
    const content = msg["content"];
    if (!Array.isArray(content)) continue;
    for (const blockRaw of content) {
      const block = _asRecord(blockRaw);
      if (block && block["type"] === "tool_use") {
        const tuId = block["id"] as string | undefined;
        const name = block["name"] as string | undefined;
        const input = block["input"] ?? {};
        if (tuId && name) pending.set(tuId, [name, input]);
      }
    }
  }

  for (const raw of messages) {
    const msg = _asRecord(raw);
    if (!msg || msg["role"] !== "user") continue;
    const content = msg["content"];
    if (!Array.isArray(content)) continue;
    for (const blockRaw of content) {
      const block = _asRecord(blockRaw);
      if (block && block["type"] === "tool_result") {
        const tuId = block["tool_use_id"] as string | undefined;
        const result = block["content"] ?? "";
        if (tuId && pending.has(tuId)) {
          const [name, args] = pending.get(tuId)!;
          _tryPopulate(name, args, result);
        }
      }
    }
  }
}

/** Scan Gemini-format contents for function_call/function_response pairs and cache. */
export function learnFromGeminiContents(contents: unknown): void {
  if (!Array.isArray(contents) || contents.length === 0) return;
  // Gemini has no per-call id in function_response; match by name (last seen).
  const pending = new Map<string, unknown>();

  for (const raw of contents) {
    const content = _asRecord(raw) ?? (raw as Record<string, unknown>);
    const role = content?.["role"] ?? (raw as { role?: string }).role;
    const parts = (content?.["parts"] ?? (raw as { parts?: unknown[] }).parts ?? []) as unknown[];

    if (role === "model") {
      for (const partRaw of parts) {
        const part = _asRecord(partRaw) ?? (partRaw as Record<string, unknown>);
        const fc = part?.["function_call"] ?? (partRaw as { function_call?: unknown }).function_call;
        const fcRec = _asRecord(fc) ?? (fc as Record<string, unknown> | null);
        if (!fcRec) continue;
        const name = fcRec["name"] as string | undefined;
        const args = fcRec["args"] ?? {};
        if (name) pending.set(name, args ?? {});
      }
    } else if (role === "tool" || role === "function" || role === "user") {
      for (const partRaw of parts) {
        const part = _asRecord(partRaw) ?? (partRaw as Record<string, unknown>);
        const fr =
          part?.["function_response"] ?? (partRaw as { function_response?: unknown }).function_response;
        const frRec = _asRecord(fr) ?? (fr as Record<string, unknown> | null);
        if (!frRec) continue;
        const name = frRec["name"] as string | undefined;
        const response = frRec["response"];
        if (name && pending.has(name)) {
          _tryPopulate(name, pending.get(name), response);
        }
      }
    }
  }
}
