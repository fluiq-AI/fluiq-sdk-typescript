/**
 * MCP (Model Context Protocol) tracing — patches @modelcontextprotocol/sdk's
 * Client so listTools() and callTool() calls land on the trace tree.
 *
 * The TS MCP SDK keeps a long-lived `Client` instance, so we capture the server
 * URL off the transport in `connect()` and stash it on the client instance —
 * no ContextVar gymnastics needed (unlike the Python SDK). All patches are
 * best-effort and no-op when the SDK is absent or not require-able.
 */
import { logTrace } from "../../tracer";

type AnyFn = (...args: unknown[]) => unknown;

interface ClientCtor {
  prototype: Record<string, unknown>;
}

function _resolveMcpClient(): ClientCtor | null {
  for (const path of [
    "@modelcontextprotocol/sdk/client/index.js",
    "@modelcontextprotocol/sdk/client/index",
    "@modelcontextprotocol/sdk",
  ]) {
    try {
      const mod = require(path) as Record<string, unknown>;
      const ctor = mod["Client"] as ClientCtor | undefined;
      if (ctor && ctor.prototype) return ctor;
    } catch {
      // try next path
    }
  }
  return null;
}

/** Extract the server URL from a transport, tolerating private/public field names. */
function _transportUrl(transport: unknown): string {
  if (transport == null || typeof transport !== "object") return "";
  const t = transport as Record<string, unknown>;
  for (const key of ["_url", "url", "_endpoint", "endpoint"]) {
    const v = t[key];
    if (v instanceof URL) return v.toString();
    if (typeof v === "string" && v) return v;
  }
  return "";
}

function _serverUrl(client: unknown): string {
  const c = client as Record<string, unknown>;
  const url = c["_fluiqServerUrl"];
  return typeof url === "string" ? url : "";
}

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

/** Patch Client.connect() to capture the server URL onto the client instance. */
export function patchMcpInitialize(): void {
  const Client = _resolveMcpClient();
  if (!Client) return;

  _patchMethod(Client.prototype, "connect", (original) =>
    async function (this: Record<string, unknown>, transport: unknown, ...rest: unknown[]) {
      const serverUrl = _transportUrl(transport);
      const result = await original.call(this, transport, ...rest);
      if (serverUrl) {
        this["_fluiqServerUrl"] = serverUrl;
      }
      return result;
    }
  );
}

/** Trace Client.listTools() calls. */
export function patchMcpListTools(): void {
  const Client = _resolveMcpClient();
  if (!Client) return;

  _patchMethod(Client.prototype, "listTools", (original) =>
    async function (this: Record<string, unknown>, ...args: unknown[]) {
      const serverUrl = _serverUrl(this);
      const result = await original.call(this, ...args);

      if (serverUrl) {
        await logTrace({
          type: "mcp",
          kind: "mcp_list_tools",
          server_url: serverUrl,
        });
      }

      return result;
    }
  );
}

/** Trace Client.callTool() invocations. */
export function patchMcpCallTool(): void {
  const Client = _resolveMcpClient();
  if (!Client) return;

  _patchMethod(Client.prototype, "callTool", (original) =>
    async function (this: Record<string, unknown>, params: unknown, ...rest: unknown[]) {
      const serverUrl = _serverUrl(this);
      const p = (params ?? {}) as Record<string, unknown>;
      const name = String(p["name"] ?? "");
      const result = await original.call(this, params, ...rest);

      if (serverUrl && name) {
        await logTrace({
          type: "mcp",
          kind: "mcp_call",
          server_url: serverUrl,
          tool_name: name,
        });
      }

      return result;
    }
  );
}
