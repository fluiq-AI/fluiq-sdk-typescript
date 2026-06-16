import type { ICache } from "../state";

/** Minimal shape of the ioredis client we depend on. */
interface RedisLike {
  get(k: string): Promise<string | null>;
  set(k: string, v: string): Promise<unknown>;
  setex(k: string, ttl: number, v: string): Promise<unknown>;
  del(k: string): Promise<number>;
  on(event: string, listener: (...args: unknown[]) => void): void;
  /** The underlying net.Socket — present once connected. */
  stream?: { ref(): void; unref(): void };
}

/**
 * Redis-backed cache using ioredis (optional peer dependency).
 * Values are serialized as JSON strings in Redis.
 *
 * The connection's socket is kept `unref`'d while idle so it never holds Node's
 * event loop open — a short script exits on its own once its work is done,
 * without any explicit shutdown call. The socket is `ref`'d only while a command
 * is actually in flight (see `_track`), so pending reads/writes — including
 * fire-and-forget cache writes — always complete before the process exits.
 */
export class RedisCache implements ICache {
  private _client: RedisLike;
  private _defaultTtl: number | undefined;
  private _prefix: string;
  private _pending = 0;

  constructor(redisUrl: string, defaultTtl?: number, prefix = "fluiq:") {
    const Redis = require("ioredis");
    this._client = new Redis(redisUrl) as RedisLike;
    this._defaultTtl = defaultTtl;
    this._prefix = prefix;
    // A fresh socket is created on every (re)connect; reapply the ref state so
    // an idle connection never keeps the event loop alive.
    this._client.on("connect", () => this._applyRef());
    // ioredis emits 'error' on connection problems; swallow so it never crashes
    // the host app (cache is best-effort).
    this._client.on("error", () => {});
  }

  private _key(key: string): string {
    return `${this._prefix}${key}`;
  }

  /** ref the socket while commands are pending, unref it once idle. */
  private _applyRef(): void {
    const stream = this._client.stream;
    if (!stream) return;
    if (this._pending > 0) stream.ref();
    else stream.unref();
  }

  /** Run a redis op with the socket ref'd for its duration so it can't be
   *  abandoned by an early process exit, then unref when the queue drains. */
  private async _track<T>(op: () => Promise<T>): Promise<T> {
    this._pending++;
    this._applyRef();
    try {
      return await op();
    } finally {
      this._pending--;
      if (this._pending < 0) this._pending = 0;
      this._applyRef();
    }
  }

  async get(key: string): Promise<unknown> {
    try {
      const raw = await this._track(() => this._client.get(this._key(key)));
      if (raw == null) return null;
      return JSON.parse(raw);
    } catch {
      return null;
    }
  }

  async set(key: string, value: unknown, ttl?: number): Promise<void> {
    try {
      const serialized = JSON.stringify(value);
      const effectiveTtl = ttl ?? this._defaultTtl;
      if (effectiveTtl && effectiveTtl > 0) {
        await this._track(() => this._client.setex(this._key(key), effectiveTtl, serialized));
      } else {
        await this._track(() => this._client.set(this._key(key), serialized));
      }
    } catch {
      // Ignore cache write failures
    }
  }

  async delete(key: string): Promise<boolean> {
    try {
      const result = await this._track(() => this._client.del(this._key(key)));
      return result > 0;
    } catch {
      return false;
    }
  }
}
