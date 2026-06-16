import { createHash } from "crypto";

function _normalize(v: unknown): unknown {
  if (v === null || v === undefined) return null;
  if (typeof v === "string" || typeof v === "number" || typeof v === "boolean") return v;
  if (Array.isArray(v)) return v.map(_normalize);
  if (typeof v === "object") {
    const obj = v as Record<string, unknown>;
    return Object.fromEntries(
      Object.keys(obj)
        .sort()
        .map((k) => [k, _normalize(obj[k])])
    );
  }
  return String(v);
}

/**
 * Stable SHA-256 over an ordered tuple of JSON-serializable parts.
 * Used by the specialized caches to derive a deterministic key from
 * (domain, model, payload, params) so the same inputs always hash to
 * the same slot — across processes and Node.js versions.
 */
export function makeKey(...parts: unknown[]): string {
  const payload = JSON.stringify(parts.map(_normalize));
  return createHash("sha256").update(payload, "utf8").digest("hex");
}
