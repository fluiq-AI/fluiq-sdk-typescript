# Changelog

All notable changes to `@fluiq/sdk`.

This project follows [Semantic Versioning](https://semver.org/).

## 0.3.0 — 2026-08-12

### Removed — breaking

The optimization pillar is gone. Fluiq now stands on three pillars: secure,
observe, evaluate, with dataset and prompt management inside evaluation. This
release mirrors Python SDK 0.3.0.

- **`fluiq.optimize()` is removed.** There is no replacement. Delete the call;
  everything else in your integration keeps working unchanged.
- **`fluiq.lookupToolResult()` is removed.** It read the tool cache, which no
  longer exists.
- `src/optimization/`, `optimizeGate.ts`, and `toolCache.ts` are removed.
- MCP `listTools()` / `callTool()` are still patched and still traced; only the
  caching layer around them is gone.
- Vector-store integrations (Chroma, Pinecone, Qdrant, Weaviate, FAISS) keep
  full tracing; their cached and cache-invalidating wrappers collapse into the
  plain traced wrappers.
- `innerCacheHit` is gone from the AsyncLocalStorage trace context.

### Migration

```diff
  import fluiq from "@fluiq/sdk";

  fluiq.instrument({ apiKey: "fl_..." });
  fluiq.secure({ mode: "block" });
- fluiq.optimize();
  fluiq.eval({ thresholds: { hallucination: 0.8 } });
```

If you called `fluiq.lookupToolResult(name, args)`, call your tool directly.

### Kept deliberately

Provider prompt-cache **token capture** stays: Anthropic
`prompt_cache_read_tokens` / `prompt_cache_creation_tokens`, and OpenAI and
Gemini `prompt_cached_tokens`. That is cost accuracy — it makes reported spend
match the provider bill — not a caching product. The SDK injects nothing and
serves nothing from a cache.

### Changed

- `ioredis` is no longer used by any code path. It was already an optional peer
  dependency; you can drop it if you installed it only for Fluiq.
- The CI eval gate endpoint moved from `GET /api/v1/optimize/evals` to
  `GET /api/v1/evaluate/recent-evals`.

## 0.2.1

Prior releases are not catalogued here.
