# Fluiq TypeScript SDK
[![npm version](https://img.shields.io/npm/v/@fluiq/sdk.svg)](https://www.npmjs.com/package/@fluiq/sdk)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Node 18+](https://img.shields.io/badge/node-18+-green.svg)](https://nodejs.org/)

> **Status: archived.** Fluiq ran from 10 April to September 2026 and never
> found customers. The hosted service is shut down and the infrastructure is
> gone. The code is MIT and stays public because it works. Nothing here is
> maintained — fork it freely.
>
> The rest of the project: [fluiq-api](https://github.com/fluiq-AI/fluiq-api) ·
> [tracer](https://github.com/fluiq-AI/fluiq-worker-tracer) ·
> [evaluator](https://github.com/fluiq-AI/fluiq-worker-evaluator) ·
> [security](https://github.com/fluiq-AI/fluiq-worker-security) ·
> [Python SDK](https://github.com/fluiq-AI/fluiq-sdk) ·
> [TypeScript SDK](https://github.com/fluiq-AI/fluiq-sdk-typescript) ·
> [guardrail-bench](https://github.com/SaurabhKumbhar24/guardrail-bench)

Instrument any LLM application in two lines. Auto-tracing for OpenAI, Anthropic, Gemini, LangChain, LangGraph, Google ADK, MCP, and all major vector stores — plus a `trace()` wrapper for everything else. Every run is cost-tracked in your dashboard. Add one more line to enable security scanning, evaluation, or Redis caching.

---

## Installation

```bash
npm install @fluiq/sdk
```

**Requires Node.js 18+.** Written in TypeScript; ships its own type definitions.

---

## Quickstart

```typescript
import fluiq from "@fluiq/sdk";

fluiq.instrument({ apiKey: "fl_..." });

// Every OpenAI / Anthropic / Gemini / LangChain / MCP call is now traced.
```

Use the `FLUIQ_API_KEY` environment variable to avoid hardcoding the key:

```typescript
import fluiq from "@fluiq/sdk";

fluiq.instrument(); // reads FLUIQ_API_KEY automatically
```

Named imports work too:

```typescript
import { instrument, trace } from "@fluiq/sdk";
```

---

## Auto-instrumentation

`instrument()` patches every supported provider it finds installed. If a provider isn't installed the corresponding patch is skipped silently — no feature flags required. Call `instrument()` once, before you create any provider clients.

### OpenAI

```typescript
import OpenAI from "openai";
import fluiq from "@fluiq/sdk";

fluiq.instrument({ apiKey: "fl_..." });

const client = new OpenAI();
await client.chat.completions.create({
  model: "gpt-4o",
  messages: [{ role: "user", content: "Hello" }],
});
```

Covers chat completions, the responses API, structured outputs (`.parse`), streaming, embeddings, images, and audio.

### Anthropic

```typescript
import Anthropic from "@anthropic-ai/sdk";
import fluiq from "@fluiq/sdk";

fluiq.instrument({ apiKey: "fl_..." });

const client = new Anthropic();
await client.messages.create({
  model: "claude-opus-4-7",
  max_tokens: 512,
  messages: [{ role: "user", content: "Hello" }],
});
```

Covers messages, streaming, `countTokens`, and beta messages.

### Gemini / Vertex AI

```typescript
import { GoogleGenAI } from "@google/genai";
import fluiq from "@fluiq/sdk";

fluiq.instrument({ apiKey: "fl_..." });

const client = new GoogleGenAI({});
await client.models.generateContent({ model: "gemini-2.5-pro", contents: "Hello" });
```

Covers `generateContent`, streaming, `countTokens`, embeddings, and Vertex AI.

### LangChain

```typescript
import { ChatOpenAI } from "@langchain/openai";
import fluiq from "@fluiq/sdk";

fluiq.instrument({ apiKey: "fl_..." });

const llm = new ChatOpenAI({ model: "gpt-4o" });
await llm.invoke("Hello");
```

Chains, agents, and retrievers all emit traces automatically.

### LangGraph

```typescript
import { StateGraph } from "@langchain/langgraph";
import fluiq from "@fluiq/sdk";

fluiq.instrument({ apiKey: "fl_..." });

const graph = new StateGraph(/* ... */)
  .addNode("planner", plannerNode)
  .addNode("tool_executor", toolNode);
const app = graph.compile();
await app.invoke({ messages: [/* ... */] });
```

Each node emits its own span. The dashboard shows one row per node so you can see which step drives cost.

### Google ADK

```typescript
import fluiq from "@fluiq/sdk";

fluiq.instrument({ apiKey: "fl_..." });

// @google/adk agent, model, and tool calls are traced automatically.
```

### Vector stores

ChromaDB, Pinecone, Qdrant, Weaviate, and FAISS queries and mutations are traced automatically after `instrument()` is called. No extra setup required.

### MCP

```typescript
import fluiq from "@fluiq/sdk";

fluiq.instrument({ apiKey: "fl_..." });

// MCP client.initialize(), listTools(), and callTool() are traced alongside
// the LLM that invokes the tool.
```

---

## Custom tracing with `trace()`

Wrap any function with `trace()` to record its inputs, outputs, latency, and errors. Sync and async functions are both supported. Nested calls preserve parent/child relationships through `AsyncLocalStorage`.

```typescript
import fluiq from "@fluiq/sdk";

fluiq.instrument({ apiKey: "fl_..." });

const retrieve = fluiq.trace(async (question: string): Promise<string[]> => {
  return vectorStore.similaritySearch(question, 4);
});

const answer = fluiq.trace(async (question: string): Promise<string> => {
  const docs = await retrieve(question); // nested span
  return llm.invoke(prompt(question, docs));
});
```

Override the span name shown in the dashboard:

```typescript
const run = fluiq.trace(
  async (question: string) => { /* ... */ },
  { name: "research_agent" }
);
```

> **Fail-open by design.** Every span emission is wrapped in a safety guard so a Fluiq error never crashes your application. Network failures, malformed payloads, and missing optional dependencies are absorbed silently.

---

## Security scanning — `fluiq.secure()`

Activate server-side security scanning. Every prompt and response is checked for PII, prompt injection, jailbreaks, skeleton-key attacks, leaked secrets, and indirect injection in tool outputs. Detection runs on the Fluiq backend — patterns are never shipped in the public SDK.

**Requires Team plan or above.**

```typescript
import fluiq from "@fluiq/sdk";

fluiq.instrument({ apiKey: "fl_..." });
fluiq.secure();                  // warn mode (default)
```

In `warn` mode (default) scanning runs after each LLM call. Security fields are written into the stored trace. HIGH-risk content is redacted before persistence. Your application is never interrupted.

In `block` mode every prompt is checked *before* the LLM API call. If an attack is detected a `FluiqSecurityError` is thrown and the LLM call is never made.

```typescript
fluiq.secure({ mode: "block" });
```

```typescript
import { FluiqSecurityError } from "@fluiq/sdk";

try {
  const response = await client.chat.completions.create(/* ... */);
} catch (e) {
  if (e instanceof FluiqSecurityError) {
    console.log(e.riskLevel);    // "high"
    console.log(e.attackTypes);  // ["jailbreak", "prompt_injection"]
  }
}
```

**Parameters**

| Parameter | Values | Default | Description |
|---|---|---|---|
| `mode` | `"warn"` \| `"block"` | `"warn"` | `warn`: post-call scan only. `block`: pre-call guard + post-call scan. |
| `guardrail` | `string` | `"default"` | Slug of a named guardrail policy configured in the dashboard. |

**What gets scanned**

| Category | Detects |
|---|---|
| PII | Credit cards, SSNs, IBANs, email, phone, IP address, names, API keys |
| Prompt injection | Instruction-override patterns, system-prompt leaking, template injection |
| Jailbreak | Role-play escapes, persona hijacks, fictional-framing bypasses, DAN and variants |
| Skeleton key | "Add a mode / unlock capabilities" style attacks |
| Secrets | OpenAI / Anthropic / AWS / GitHub / Stripe key patterns, high-entropy tokens |
| Indirect injection | Injection patterns in tool outputs and retrieved context documents |

Security findings are visible on the **Security** tab in the Traces drawer.

---

## Evaluation — `fluiq.eval()`

Activate server-side LLM-as-judge evaluation. After each LLM call Fluiq scores the response on the requested metrics (0 = worst, 1 = best) and stores the results in your dashboard.

```typescript
import fluiq from "@fluiq/sdk";

fluiq.instrument({ apiKey: "fl_..." });
fluiq.eval();                    // warn mode, default metrics
```

In `warn` mode (default) evaluation runs in the background and logs a warning when a score falls below its threshold. Your application is never interrupted.

In `block` mode evaluation runs synchronously after each LLM call. If any metric falls below its threshold a `FluiqEvalError` is thrown.

```typescript
fluiq.eval({
  mode: "block",
  thresholds: { hallucination: 0.8, relevance: 0.7 },
  metrics: ["hallucination", "relevance", "toxicity"],
  judgeModel: "claude-haiku-4-5-20251001",
});
```

```typescript
import { FluiqEvalError } from "@fluiq/sdk";

try {
  const response = await client.chat.completions.create(/* ... */);
} catch (e) {
  if (e instanceof FluiqEvalError) {
    console.log(e.failures);  // { hallucination: 0.61 }
    console.log(e.scores);    // { hallucination: 0.61, relevance: 0.94 }
  }
}
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mode` | `"warn"` \| `"block"` | `"warn"` | `warn`: background eval + log. `block`: synchronous eval + throw on failure. |
| `metrics` | `string[]` | `["hallucination", "relevance"]` | Which metrics to score. |
| `thresholds` | `Record<string, number>` | `{}` | Per-metric pass/fail thresholds (0–1). Warnings / errors only fire when a threshold is set. |
| `judgeModel` | `string` | `"claude-haiku-4-5-20251001"` | The model Fluiq uses as judge. |

**Supported metrics**

`hallucination`, `faithfulness`, `relevance`, `toxicity`, `coherence`, `completeness`

---

## Combining features

All three features compose freely:

```typescript
import fluiq from "@fluiq/sdk";

fluiq.instrument({ apiKey: "fl_..." });
fluiq.secure({ mode: "block" });
fluiq.eval({ thresholds: { hallucination: 0.8 }, mode: "warn" });
```

Call order per LLM request:
1. **secure (block)** — pre-call prompt check; throws `FluiqSecurityError` if blocked
2. LLM API call
3. **secure (warn)** — post-call scan; enriches trace with security fields
4. **eval** — evaluation in the background (warn) or synchronously (block)

---

## Configuration reference

```typescript
fluiq.instrument({
  apiKey?: string,    // defaults to FLUIQ_API_KEY env var
  endpoint?: string,  // defaults to https://api.getfluiq.com/api
  version?: string,   // defaults to "v1"
});
```

| Parameter | Default | Description |
|---|---|---|
| `apiKey` | `FLUIQ_API_KEY` env var | Your workspace API key. |
| `endpoint` | `https://api.getfluiq.com/api` | Ingest URL. Override for self-hosted deployments. Set via `FLUIQ_API_ENDPOINT`. |
| `version` | `"v1"` | Trace schema version. Pin in production to opt in to schema bumps explicitly. |

**Environment variables**

| Variable | Description |
|---|---|
| `FLUIQ_API_KEY` | Default API key used by `instrument()`. |
| `FLUIQ_API_ENDPOINT` | Default endpoint URL. |

Traces are sent fire-and-forget; in-flight requests are drained automatically on `beforeExit`, so short-lived scripts deliver their final spans without any manual flush.

---

## Exceptions

| Exception | Thrown when |
|---|---|
| `FluiqSecurityError` | `fluiq.secure({ mode: "block" })` is active and the pre-call check returns a HIGH-risk result. Exposes `riskLevel`, `attackTypes`, `blockReason`. |
| `FluiqEvalError` | `fluiq.eval({ mode: "block" })` is active and one or more metrics fall below their threshold. Exposes `failures`, `scores`. |

Both are importable from `@fluiq/sdk`:

```typescript
import { FluiqSecurityError, FluiqEvalError } from "@fluiq/sdk";
```

---

## Plan requirements

| Feature | Minimum plan |
|---|---|
| `fluiq.instrument()` | Free |
| `fluiq.trace()` | Free |
| `fluiq.eval()` | Free |
| `fluiq.secure()` | Free |

Metered features degrade gracefully: when a plan limit is reached the SDK falls back to no-op behaviour automatically — your application continues to run unaffected.

---

## License

MIT. See [LICENSE](LICENSE).
