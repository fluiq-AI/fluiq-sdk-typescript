/**
 * Gemini integration — patches @google/genai Models.generateContent (sync + async)
 * and Models.generateContentStream. Silently skipped if the package is not installed.
 */
import { randomUUID } from "crypto";
import { logTrace } from "../tracer";
import { _config } from "../config";
import { TraceType, toPlainObject, type TraceTypeValue } from "./shared/models";
import {
  currentParentId,
  withLlmTraceId,
  runInRootContext,
  isInLangchainLlm,
} from "./shared/context";
import { preCallGuard } from "./shared/securityGate";
import { FluiqSecurityError } from "../exceptions";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function _toJsonable(obj: unknown): unknown {
  return _toJsonableInner(obj, new WeakSet<object>(), 0);
}

// Cycle- and depth-guarded so non-plain values (e.g. a live MCP Client passed
// as a tool, which has circular internals) can't overflow the stack. The SDK
// must never crash the host application while building a trace.
function _toJsonableInner(obj: unknown, seen: WeakSet<object>, depth: number): unknown {
  if (obj == null) return null;
  if (typeof obj !== "object") return obj;
  if (seen.has(obj as object)) return "[Circular]";
  if (depth > 8) return "[Truncated]";
  seen.add(obj as object);
  try {
    if (Array.isArray(obj)) return obj.map((v) => _toJsonableInner(v, seen, depth + 1));
    const o = obj as Record<string, unknown>;
    return Object.fromEntries(
      Object.entries(o).map(([k, v]) => [k, _toJsonableInner(v, seen, depth + 1)])
    );
  } finally {
    seen.delete(obj as object);
  }
}

/** An MCP Client passed directly as a tool — don't serialize its internals. */
function _looksLikeMcpClient(t: unknown): boolean {
  return (
    !!t &&
    typeof t === "object" &&
    typeof (t as Record<string, unknown>)["callTool"] === "function" &&
    typeof (t as Record<string, unknown>)["listTools"] === "function"
  );
}

/** Serialize request tools, summarizing any live MCP Client (avoids cycles + secret leakage). */
function _sanitizeTools(tools: unknown): unknown {
  if (!Array.isArray(tools)) return _toJsonable(tools);
  return tools.map((t) =>
    _looksLikeMcpClient(t)
      ? { type: "mcp_client", server_url: (t as Record<string, unknown>)["_fluiqServerUrl"] ?? null }
      : _toJsonable(t)
  );
}

/** Read the first defined key off an object — tolerates snake_case (REST/Python) and camelCase (JS SDK). */
function _g(obj: unknown, ...keys: string[]): unknown {
  if (obj == null || typeof obj !== "object") return undefined;
  const o = obj as Record<string, unknown>;
  for (const k of keys) {
    const v = o[k];
    if (v !== undefined && v !== null) return v;
  }
  return undefined;
}

function _candidateParts(cand: unknown): unknown[] {
  const content = _g(cand, "content");
  const parts = _g(content, "parts");
  return Array.isArray(parts) ? parts : [];
}

const MEDIA_TYPES = new Set(["image", "video", "audio", "file_data", "inline_data"]);

function _stripPart(part: unknown): unknown | null {
  if (typeof part !== "object" || part === null) return part;
  const p = part as Record<string, unknown>;
  for (const t of MEDIA_TYPES) {
    if (t in p) return null;
  }
  return _toJsonable(p);
}

function _stripContent(content: unknown): unknown {
  if (!Array.isArray(content)) return _toJsonable(content);
  const kept: unknown[] = [];
  for (const part of content) {
    const stripped = _stripPart(part);
    if (stripped !== null) kept.push(stripped);
  }
  return kept.length > 0 ? kept : null;
}

function _flattenCandidatesText(candidates: unknown): string | null {
  if (!Array.isArray(candidates)) return null;
  const parts: string[] = [];
  for (const cand of candidates) {
    for (const part of _candidateParts(cand)) {
      const p = part as Record<string, unknown>;
      // Skip thought parts — they belong in `thinking`, not the response text.
      if (p["thought"] === true) continue;
      if (typeof p["text"] === "string" && p["text"]) parts.push(p["text"]);
    }
  }
  return parts.length > 0 ? parts.join("\n") : null;
}

function _extractFunctionCalls(candidates: unknown): unknown[] | null {
  if (!Array.isArray(candidates)) return null;
  const calls: unknown[] = [];
  for (const cand of candidates) {
    for (const part of _candidateParts(cand)) {
      const fc = _g(part, "function_call", "functionCall");
      if (fc) calls.push(_toJsonable(fc));
    }
  }
  return calls.length > 0 ? calls : null;
}

function _finishReasons(candidates: unknown): unknown[] | null {
  if (!Array.isArray(candidates)) return null;
  const reasons = candidates.map((cand) => {
    const r = _g(cand, "finish_reason", "finishReason");
    return r != null ? String(r) : null;
  });
  return reasons.length > 0 ? reasons : null;
}

function _extractThinking(candidates: unknown): unknown[] | null {
  if (!Array.isArray(candidates)) return null;
  const thoughts: unknown[] = [];
  for (const cand of candidates) {
    for (const part of _candidateParts(cand)) {
      const p = part as Record<string, unknown>;
      if (p["thought"]) {
        thoughts.push({
          text: p["text"] ?? null,
          signature: _g(p, "thought_signature", "thoughtSignature") ?? null,
        });
      }
    }
  }
  return thoughts.length > 0 ? thoughts : null;
}

function _extractUsage(response: Record<string, unknown>): Record<string, number | null> | null {
  const usage = _g(response, "usage_metadata", "usageMetadata");
  if (!usage) return null;
  const prompt = _g(usage, "prompt_token_count", "promptTokenCount");
  const completion = _g(usage, "candidates_token_count", "candidatesTokenCount");
  const total = _g(usage, "total_token_count", "totalTokenCount");
  return {
    prompt: prompt != null ? Number(prompt) : null,
    completion: completion != null ? Number(completion) : null,
    total: total != null ? Number(total) : null,
  };
}

function _cachedContentTokens(response: Record<string, unknown>): number | null {
  const usage = _g(response, "usage_metadata", "usageMetadata");
  const v = _g(usage, "cached_content_token_count", "cachedContentTokenCount");
  return typeof v === "number" ? v : null;
}

function _systemInstruction(params: Record<string, unknown>): unknown {
  const config = params["config"];
  return _toJsonable(_g(config, "system_instruction", "systemInstruction"));
}

function _extractRequestTools(params: Record<string, unknown>): { tools: unknown; toolConfig: unknown } {
  const config = params["config"] as Record<string, unknown> | undefined;
  const tools = params["tools"] ?? config?.["tools"] ?? null;
  const toolConfig =
    params["tool_config"] ?? params["toolConfig"] ?? config?.["tool_config"] ?? config?.["toolConfig"] ?? null;
  return { tools: _sanitizeTools(tools), toolConfig: _toJsonable(toolConfig) };
}

/** Extract MCP server descriptors declared in request tools (config.tools[].mcp_servers). */
function _extractMcpServers(params: Record<string, unknown>): unknown[] | null {
  const config = params["config"] as Record<string, unknown> | undefined;
  const tools = (params["tools"] ?? config?.["tools"]) as unknown;
  if (!Array.isArray(tools)) return null;
  const found: unknown[] = [];
  for (const tool of tools) {
    const mcp = _g(tool, "mcp_servers", "mcpServers");
    if (Array.isArray(mcp)) {
      for (const srv of mcp) found.push(_toJsonable(srv));
    }
  }
  return found.length > 0 ? found : null;
}

function _getModel(params: Record<string, unknown>): string | null {
  return (params["model"] as string | null) ?? null;
}

// ---------------------------------------------------------------------------
// Pending tool call latency tracking
// ---------------------------------------------------------------------------

const _pendingToolCalls = new Map<string, { ts: number; name: string | null }>();

function _recordDispatchedToolCalls(candidates: unknown): void {
  if (!Array.isArray(candidates)) return;
  const now = Date.now() / 1000;
  for (const cand of candidates) {
    for (const part of _candidateParts(cand)) {
      const fc = _g(part, "function_call", "functionCall") as Record<string, unknown> | undefined;
      if (!fc) continue;
      const id = String(fc["id"] ?? fc["name"] ?? randomUUID());
      const name = typeof fc["name"] === "string" ? fc["name"] : null;
      _pendingToolCalls.set(id, { ts: now, name });
    }
  }
}

function _gcPendingToolCalls(ttl = 3600): void {
  const cutoff = Date.now() / 1000 - ttl;
  for (const [id, { ts }] of _pendingToolCalls) {
    if (ts < cutoff) _pendingToolCalls.delete(id);
  }
}

function _computeToolCallLatencies(contents: unknown): unknown[] | null {
  if (!Array.isArray(contents)) return null;
  const now = Date.now() / 1000;
  const latencies: unknown[] = [];
  for (const content of contents) {
    if (typeof content !== "object" || content === null) continue;
    const c = content as Record<string, unknown>;
    const parts = c["parts"];
    if (!Array.isArray(parts)) continue;
    for (const part of parts) {
      const fr = _g(part, "function_response", "functionResponse") as Record<string, unknown> | undefined;
      if (!fr) continue;
      const name = String(fr["name"] ?? "");
      const entry = [..._pendingToolCalls.entries()].find(
        ([, v]) => v.name === name
      );
      if (!entry) continue;
      _pendingToolCalls.delete(entry[0]);
      latencies.push({ tool_call_id: entry[0], name, latency: now - entry[1].ts });
    }
  }
  return latencies.length > 0 ? latencies : null;
}

// ---------------------------------------------------------------------------
// Emit helpers
// ---------------------------------------------------------------------------

async function _emitGeminiTrace(
  params: Record<string, unknown>,
  response: Record<string, unknown>,
  start: number,
  end: number,
  toolCallLatencies: unknown
): Promise<void> {
  const candidates = response["candidates"];
  _recordDispatchedToolCalls(candidates);
  const text = _flattenCandidatesText(candidates);
  const functionCalls = _extractFunctionCalls(candidates);
  const thinking = _extractThinking(candidates);
  const tokens = _extractUsage(response);
  const { tools, toolConfig } = _extractRequestTools(params);

  await logTrace(
    toPlainObject({
      type: "llm",
      integration: TraceType.Gemini,
      model: _getModel(params),
      contents: _toJsonable(params["contents"]),
      system_instruction: _systemInstruction(params),
      tools,
      tool_config: toolConfig,
      response: text,
      thinking,
      mcp_servers: _extractMcpServers(params),
      function_calls: functionCalls,
      tool_call_latencies: toolCallLatencies,
      finish_reasons: _finishReasons(candidates),
      latency: end - start,
      parent_id: currentParentId(),
      tokens: tokens
        ? { prompt: tokens.prompt, completion: tokens.completion, total: tokens.total }
        : null,
      prompt_cached_tokens: _cachedContentTokens(response),
    })
  );
}

async function _emitGeminiError(
  params: Record<string, unknown>,
  error: Error,
  start: number,
  end: number,
  api?: string
): Promise<void> {
  await logTrace(
    toPlainObject({
      type: "llm",
      integration: TraceType.Gemini,
      api: api ?? null,
      model: _getModel(params),
      contents: _toJsonable(params["contents"]),
      output: String(error),
      latency: end - start,
      parent_id: currentParentId(),
      success: false,
    })
  );
}

async function _emitSecurityBlockedTrace(
  params: Record<string, unknown>,
  exc: FluiqSecurityError,
  start: number,
  end: number
): Promise<void> {
  const attackTypes = exc.attackTypes;
  const has = (t: string) => attackTypes.includes(t);
  await logTrace({
    type: "llm",
    integration: TraceType.Gemini,
    api: "generate_content",
    model: _getModel(params),
    contents: _toJsonable(params["contents"]),
    latency: end - start,
    parent_id: currentParentId(),
    success: false,
    _security_pre_blocked: true,
    status: "blocked",
    security_risk_level: exc.riskLevel,
    security_risk_score: 1.0,
    should_block: true,
    block_reason: exc.blockReason,
    injection_detected: has("prompt_injection"),
    injection_patterns: has("prompt_injection") ? ["prompt_injection"] : [],
    jailbreak_detected: has("jailbreak"),
    jailbreak_patterns: has("jailbreak") ? ["jailbreak"] : [],
    skeleton_key_detected: has("skeleton_key"),
    skeleton_key_patterns: has("skeleton_key") ? ["skeleton_key"] : [],
    secrets_detected: false,
    secret_types: [],
    pii_entities_prompt: [],
    pii_entities_response: [],
    prompt_redacted: "",
    response_redacted: "",
    indirect_injection_detected: false,
    indirect_injection_sources: [],
    semantic_attack_score: 0.0,
  });
}

// ---------------------------------------------------------------------------
// Patch: @google/genai Models.generateContent
// ---------------------------------------------------------------------------

export function patchGemini(): void {
  const Models = _resolveGenaiModels();
  if (!Models) return;
  _patchModelsPrototype(Models);
}

/**
 * Resolve the prototype method name to patch. In @google/genai >=0.3 the public
 * `generateContent` / `generateContentStream` are per-instance arrow fields that
 * delegate to the prototype methods `*Internal`, so they are not on the prototype
 * to patch. Prefer the public method when a version exposes it on the prototype,
 * otherwise patch the internal delegate (which the public field calls).
 */
function _resolveGenerateName(
  proto: Record<string, unknown>,
  publicName: string,
  internalName: string
): string | null {
  if (typeof proto[publicName] === "function") return publicName;
  if (typeof proto[internalName] === "function") return internalName;
  return null;
}

function _patchModelsPrototype(Models: { prototype: Record<string, unknown> }): void {
  const generateName = _resolveGenerateName(
    Models.prototype,
    "generateContent",
    "generateContentInternal"
  );
  const originalGenerate = (generateName ? Models.prototype[generateName] : undefined) as
    | ((this: unknown, params: unknown) => Promise<unknown>)
    | undefined;

  if (
    originalGenerate &&
    generateName &&
    !(originalGenerate as unknown as Record<string, boolean>)["_fluiq_patched"]
  ) {
    async function wrappedGenerateContent(
      this: unknown,
      params: Record<string, unknown>
    ): Promise<unknown> {
      if (isInLangchainLlm()) return originalGenerate!.call(this, params);
      return runInRootContext(async () => {
        _gcPendingToolCalls();
        const toolCallLatencies = _computeToolCallLatencies(
          params["contents"] as unknown[] | undefined
        );
        const traceId = randomUUID();
        const start = Date.now() / 1000;

        await logTrace(
          toPlainObject({
            type: "llm",
            integration: TraceType.Gemini,
            api: "generate_content",
            trace_id: traceId,
            model: _getModel(params),
            contents: _toJsonable(params["contents"]),
            status: "running",
            started_at: start,
            parent_id: currentParentId(),
          })
        );

        return withLlmTraceId(traceId, async () => {
          try {
            await preCallGuard(params);
          } catch (secExc) {
            if (secExc instanceof FluiqSecurityError) {
              await _emitSecurityBlockedTrace(params, secExc, start, Date.now() / 1000);
            }
            throw secExc;
          }

          let response: unknown;
          try {
            response = await originalGenerate!.call(this, params);
          } catch (e) {
            await _emitGeminiError(params, e as Error, start, Date.now() / 1000);
            throw e;
          }

          const end = Date.now() / 1000;
          await _emitGeminiTrace(
            params,
            response as Record<string, unknown>,
            start,
            end,
            toolCallLatencies
          );
          return response;
        });
      });
    }

    (wrappedGenerateContent as unknown as Record<string, boolean>)["_fluiq_patched"] = true;
    Models.prototype[generateName] = wrappedGenerateContent;
  }

  // Patch generateContentStream (or its *Internal delegate — see _resolveGenerateName)
  const streamName = _resolveGenerateName(
    Models.prototype,
    "generateContentStream",
    "generateContentStreamInternal"
  );
  const originalStream = (streamName ? Models.prototype[streamName] : undefined) as
    | ((this: unknown, params: unknown) => AsyncIterable<unknown> | Promise<AsyncIterable<unknown>>)
    | undefined;

  if (
    originalStream &&
    streamName &&
    !(originalStream as unknown as Record<string, boolean>)["_fluiq_patched"]
  ) {
    function wrappedGenerateContentStream(
      this: unknown,
      params: Record<string, unknown>
    ): AsyncIterable<unknown> | Promise<AsyncIterable<unknown>> {
      if (isInLangchainLlm()) return originalStream!.call(this, params);
      const self = this;
      const traceId = randomUUID();
      const start = Date.now() / 1000;

      async function* generator(): AsyncGenerator<unknown> {
        _gcPendingToolCalls();
        await logTrace(
          toPlainObject({
            type: "llm",
            integration: TraceType.Gemini,
            api: "generate_content_stream",
            trace_id: traceId,
            model: _getModel(params),
            contents: _toJsonable(params["contents"]),
            status: "running",
            started_at: start,
            parent_id: currentParentId(),
          })
        );

        try {
          await preCallGuard(params);
        } catch (secExc) {
          if (secExc instanceof FluiqSecurityError) {
            await _emitSecurityBlockedTrace(params, secExc as FluiqSecurityError, start, Date.now() / 1000);
          }
          throw secExc;
        }

        const toolCallLatencies = _computeToolCallLatencies(params["contents"]);
        // The public field delegates to an async `*Internal` that returns a
        // Promise<AsyncIterable>; older versions return the iterable directly.
        // `await` handles both (awaiting a non-thenable is a no-op).
        const stream = (await originalStream!.call(self, params)) as AsyncIterable<unknown>;
        const needsGate = _config.secure && _config.secure_mode === "block";
        let latest: unknown = null;

        // Secure-block: buffer the whole stream so the response gate can throw
        // before any chunk reaches the caller (mirrors the Python passthrough).
        if (needsGate) {
          const buffered: unknown[] = [];
          try {
            for await (const chunk of stream) {
              latest = chunk;
              buffered.push(chunk);
            }
          } catch (e) {
            await _emitGeminiError(params, e as Error, start, Date.now() / 1000, "generate_content_stream");
            throw e;
          }
          if (latest != null) {
            await withLlmTraceId(traceId, () =>
              _emitGeminiTrace(params, latest as Record<string, unknown>, start, Date.now() / 1000, toolCallLatencies)
            );
          }
          for (const chunk of buffered) yield chunk;
          return;
        }

        try {
          for await (const chunk of stream) {
            latest = chunk;
            yield chunk;
          }
        } catch (e) {
          await _emitGeminiError(params, e as Error, start, Date.now() / 1000, "generate_content_stream");
          throw e;
        }

        if (latest != null) {
          const end = Date.now() / 1000;
          await withLlmTraceId(traceId, () =>
            _emitGeminiTrace(
              params,
              latest as Record<string, unknown>,
              start,
              end,
              toolCallLatencies
            )
          );
        }
      }

      return generator();
    }

    (wrappedGenerateContentStream as unknown as Record<string, boolean>)["_fluiq_patched"] = true;
    Models.prototype[streamName] = wrappedGenerateContentStream;
  }
}


// ---------------------------------------------------------------------------
// Models resolver (shared across genai patches)
// ---------------------------------------------------------------------------

function _resolveGenaiModels(): { prototype: Record<string, unknown> } | null {
  const paths = ["@google/genai", "@google/genai/build/src/models"];
  for (const p of paths) {
    try {
      const m = require(p) as Record<string, unknown>;
      const Models = m["Models"] as { prototype?: Record<string, unknown> } | undefined;
      if (Models && Models.prototype) return Models as { prototype: Record<string, unknown> };
    } catch {
      /* try next path */
    }
  }
  return null;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyFn = (this: unknown, ...args: any[]) => unknown;

function _patchMethod(
  proto: Record<string, unknown>,
  method: string,
  wrap: (orig: AnyFn) => AnyFn
): void {
  const orig = proto[method];
  if (typeof orig !== "function") return;
  if ((orig as unknown as Record<string, boolean>)["_fluiq_patched"]) return;
  const wrapped = wrap(orig as AnyFn);
  (wrapped as unknown as Record<string, boolean>)["_fluiq_patched"] = true;
  proto[method] = wrapped;
}

// ---------------------------------------------------------------------------
// count_tokens
// ---------------------------------------------------------------------------

async function _emitCountTokensTrace(
  integration: TraceTypeValue,
  params: Record<string, unknown>,
  contents: unknown,
  response: unknown,
  start: number,
  end: number,
  model?: string | null
): Promise<void> {
  const total = _g(response, "total_tokens", "totalTokens");
  await logTrace(
    toPlainObject({
      type: "llm",
      integration,
      api: "count_tokens",
      model: model ?? _getModel(params),
      contents: _toJsonable(contents),
      response: _toJsonable(response),
      latency: end - start,
      parent_id: currentParentId(),
      tokens:
        total != null ? { prompt: Number(total), completion: null, total: Number(total) } : null,
    })
  );
}

export function patchGeminiCountTokens(): void {
  const Models = _resolveGenaiModels();
  if (!Models) return;
  _patchMethod(Models.prototype, "countTokens", (original) =>
    async function (this: unknown, params: Record<string, unknown>) {
      const start = Date.now() / 1000;
      let response: unknown;
      try {
        response = await original.call(this, params);
      } catch (e) {
        await _emitGeminiError(params, e as Error, start, Date.now() / 1000, "count_tokens");
        throw e;
      }
      await _emitCountTokensTrace(TraceType.Gemini, params, params["contents"], response, start, Date.now() / 1000);
      return response;
    }
  );
}

// ---------------------------------------------------------------------------
// embeddings (embedContent)
// ---------------------------------------------------------------------------

function _serializeEmbeddings(response: unknown): unknown {
  const embeddings = _g(response, "embeddings") as unknown[] | undefined;
  if (!Array.isArray(embeddings)) return _toJsonable(response);
  return {
    data: embeddings.map((emb, i) => ({ values: _g(emb, "values") ?? null, index: i })),
  };
}

export function patchGeminiEmbeddings(): void {
  const Models = _resolveGenaiModels();
  if (!Models) return;
  _patchMethod(Models.prototype, "embedContent", (original) =>
    async function (this: unknown, params: Record<string, unknown>) {
      const start = Date.now() / 1000;
      const response = await original.call(this, params);
      await logTrace(
        toPlainObject({
          type: "llm",
          integration: TraceType.Gemini,
          api: "embeddings",
          model: _getModel(params),
          contents: _toJsonable(params["contents"]),
          response: _serializeEmbeddings(response),
          latency: Date.now() / 1000 - start,
          parent_id: currentParentId(),
          tokens: null,
        })
      );
      return response;
    }
  );
}

// ---------------------------------------------------------------------------
// Vertex AI (@google-cloud/vertexai GenerativeModel) — best-effort
// ---------------------------------------------------------------------------

function _vertexModel(self: unknown): string | null {
  const m = _g(self, "_model_name", "model_name", "model", "publisherModelEndpoint");
  return typeof m === "string" ? m : null;
}

export function patchGeminiVertex(): void {
  let GenerativeModel: { prototype: Record<string, unknown> } | null = null;
  try {
    const mod = require("@google-cloud/vertexai") as Record<string, unknown>;
    GenerativeModel = mod["GenerativeModel"] as { prototype: Record<string, unknown> } | null;
  } catch {
    return;
  }
  if (!GenerativeModel || !GenerativeModel.prototype) return;

  _patchMethod(GenerativeModel.prototype, "generateContent", (original) =>
    async function (this: unknown, request: unknown) {
      if (isInLangchainLlm()) return original.call(this, request);
      return runInRootContext(async () => {
        _gcPendingToolCalls();
        const model = _vertexModel(this);
        const params: Record<string, unknown> = {
          model,
          contents: _g(request, "contents") ?? request,
        };
        const start = Date.now() / 1000;
        const traceId = randomUUID();
        await logTrace(
          toPlainObject({
            type: "llm",
            integration: TraceType.Gemini,
            api: "vertex.generate_content",
            trace_id: traceId,
            model,
            contents: _toJsonable(params["contents"]),
            status: "running",
            started_at: start,
            parent_id: currentParentId(),
          })
        );
        return withLlmTraceId(traceId, async () => {
          let result: unknown;
          try {
            result = await original.call(this, request);
          } catch (e) {
            await _emitGeminiError(params, e as Error, start, Date.now() / 1000, "vertex.generate_content");
            throw e;
          }
          // Vertex wraps the payload as { response: GenerateContentResponse }.
          const response = (_g(result, "response") ?? result) as Record<string, unknown>;
          await _emitGeminiTrace(params, response, start, Date.now() / 1000, _computeToolCallLatencies(params["contents"]));
          return result;
        });
      });
    }
  );

  _patchMethod(GenerativeModel.prototype, "countTokens", (original) =>
    async function (this: unknown, request: unknown) {
      const model = _vertexModel(this);
      const contents = _g(request, "contents") ?? request;
      const start = Date.now() / 1000;
      let response: unknown;
      try {
        response = await original.call(this, request);
      } catch (e) {
        await _emitGeminiError({ model, contents }, e as Error, start, Date.now() / 1000, "count_tokens");
        throw e;
      }
      await _emitCountTokensTrace(TraceType.Gemini, { model }, contents, response, start, Date.now() / 1000, model);
      return response;
    }
  );
}
