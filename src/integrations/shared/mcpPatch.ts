/**
 * MCP (Model Context Protocol) caching — patches @modelcontextprotocol/sdk's
 * Client so that list_tools() and call_tool() results are served from the
 * trace-driven Redis cache when `fluiq.optimize()` is active.
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

/**
 * MCP caching is a feature of `fluiq.optimize()`. When optimize is off we must
 * not touch the cache layer — doing so lazily fetches the optimize profile and
 * opens a Redis connection, which surprises callers who only use MCP for traces.
 */
function _optimizeOn(): boolean {
  try {
    return (require("../../config") as typeof import("../../config"))._config.optimize;
  } catch {
    return false;
  }
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

/**
 * Patch Client.connect() to capture the server URL onto the client instance.
 * A (re)connect means the tool list may have changed, so we invalidate the
 * cached list_tools() for that server.
 */
export function patchMcpInitialize(): void {
  const Client = _resolveMcpClient();
  if (!Client) return;

  _patchMethod(Client.prototype, "connect", (original) =>
    async function (this: Record<string, unknown>, transport: unknown, ...rest: unknown[]) {
      const serverUrl = _transportUrl(transport);
      const result = await original.call(this, transport, ...rest);
      if (serverUrl && _optimizeOn()) {
        this["_fluiqServerUrl"] = serverUrl;
        try {
          const { invalidateMcpToolsCache } =
            require("../../optimization/client") as typeof import("../../optimization/client");
          await invalidateMcpToolsCache(serverUrl);
        } catch {
          // ignore
        }
      }
      return result;
    }
  );
}

/** Cache Client.listTools() responses in Redis, keyed by server URL. */
export function patchMcpListTools(): void {
  const Client = _resolveMcpClient();
  if (!Client) return;

  _patchMethod(Client.prototype, "listTools", (original) =>
    async function (this: Record<string, unknown>, ...args: unknown[]) {
      if (!_optimizeOn()) return original.call(this, ...args);
      const serverUrl = _serverUrl(this);
      const opt = require("../../optimization/client") as typeof import("../../optimization/client");

      if (serverUrl) {
        const cachedTools = await opt.lookupMcpToolsCache(serverUrl);
        if (cachedTools != null) {
          await logTrace({
            type: "mcp",
            kind: "mcp_list_tools",
            server_url: serverUrl,
            cache_hit: true,
          });
          return { tools: cachedTools };
        }
      }

      const result = (await original.call(this, ...args)) as Record<string, unknown>;

      if (serverUrl) {
        const tools = (result?.["tools"] as unknown[]) ?? [];
        await opt.populateMcpToolsCache(serverUrl, tools);
        await logTrace({
          type: "mcp",
          kind: "mcp_list_tools",
          server_url: serverUrl,
          cache_hit: false,
        });
      }

      return result;
    }
  );
}

/** Cache Client.callTool() results in Redis, keyed by (server URL, tool name, args). */
export function patchMcpCallTool(): void {
  const Client = _resolveMcpClient();
  if (!Client) return;

  _patchMethod(Client.prototype, "callTool", (original) =>
    async function (this: Record<string, unknown>, params: unknown, ...rest: unknown[]) {
      if (!_optimizeOn()) return original.call(this, params, ...rest);
      const serverUrl = _serverUrl(this);
      const p = (params ?? {}) as Record<string, unknown>;
      const name = String(p["name"] ?? "");
      const argsDict = (p["arguments"] as Record<string, unknown>) ?? {};
      const opt = require("../../optimization/client") as typeof import("../../optimization/client");

      if (serverUrl && name) {
        const cached = await opt.lookupMcpCallCache(serverUrl, name, argsDict);
        if (cached != null) {
          await logTrace({
            type: "mcp",
            kind: "mcp_call",
            server_url: serverUrl,
            tool_name: name,
            cache_hit: true,
          });
          return { content: cached["content"] ?? [], isError: cached["isError"] ?? false };
        }
      }

      const result = (await original.call(this, params, ...rest)) as Record<string, unknown>;

      if (serverUrl && name && !result?.["isError"]) {
        const content = (result?.["content"] as unknown[]) ?? [];
        await opt.populateMcpCallCache(serverUrl, name, argsDict, content, false);
        await logTrace({
          type: "mcp",
          kind: "mcp_call",
          server_url: serverUrl,
          tool_name: name,
          cache_hit: false,
        });
      }

      return result;
    }
  );
}
