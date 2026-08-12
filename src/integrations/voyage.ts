/**
 * Voyage AI integration — patches the `voyageai` package's VoyageAIClient.embed.
 * Silently skipped if the package is not installed.
 *
 * The official TypeScript SDK calls `client.embed({ input, model })` and returns
 * `{ data: [{ embedding, index }], ... }`, whereas the Python SDK calls
 * `client.embed(texts, model)` and returns `.embeddings`. We tolerate both
 * response shapes when serializing.
 */
import { logTrace } from "../tracer";
import { TraceType } from "./shared/models";
import { currentParentId } from "./shared/context";

interface VoyageClientCtor {
  prototype: Record<string, unknown>;
}

type AnyFn = (...args: unknown[]) => unknown;

function _resolveVoyageClient(): VoyageClientCtor | null {
  let mod: Record<string, unknown>;
  try {
    mod = require("voyageai") as Record<string, unknown>;
  } catch {
    return null;
  }
  // Fern-generated SDK exports `VoyageAIClient`; older builds used `Client`.
  const ctor = (mod["VoyageAIClient"] ?? mod["Client"]) as VoyageClientCtor | undefined;
  if (!ctor || !ctor.prototype) return null;
  return ctor;
}

/** Normalise a Voyage embeddings response into `{ data: [{ values, index }] }`. */
function _serializeEmbeddings(result: unknown): { data: Array<{ values: unknown; index: number }> } {
  const r = (result ?? {}) as Record<string, unknown>;
  // TS SDK: result.data = [{ embedding, index }]
  const data = r["data"];
  if (Array.isArray(data)) {
    return {
      data: data.map((item, i) => {
        const it = (item ?? {}) as Record<string, unknown>;
        return { values: it["embedding"] ?? it["values"] ?? null, index: (it["index"] as number) ?? i };
      }),
    };
  }
  // Python-style SDK: result.embeddings = [[...], [...]]
  const embeddings = r["embeddings"];
  if (Array.isArray(embeddings)) {
    return { data: embeddings.map((emb, i) => ({ values: emb, index: i })) };
  }
  return { data: [] };
}

/** Rebuild a Voyage-shaped response object from a cached payload. */

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

export function patchVoyage(): void {
  const Client = _resolveVoyageClient();
  if (!Client) return;

  _patchMethod(Client.prototype, "embed", (original) =>
    async function (this: unknown, request: unknown, ...rest: unknown[]) {
      // TS SDK: embed({ input, model }). Python-style: embed(texts, model).
      const req = (request ?? {}) as Record<string, unknown>;
      const input = req["input"] ?? req["texts"] ?? request;
      const model = (req["model"] ?? rest[0] ?? "") as string;

      const start = Date.now() / 1000;
      const result = await original.call(this, request, ...rest);
      const serialized = _serializeEmbeddings(result);
      await logTrace({
        type: "llm",
        integration: TraceType.Voyage,
        api: "embeddings",
        model,
        input,
        response: serialized,
        latency: Date.now() / 1000 - start,
        parent_id: currentParentId(),
        tokens: null,
      });
      return result;
    }
  );
}
