export interface OptimizeProfile {
  redis_url?: string;
  key_prefix?: string;
  models?: string[];
  ttl_seconds?: number;
}

export interface ICache {
  get(key: string): Promise<unknown>;
  set(key: string, value: unknown, ttl?: number): Promise<void>;
  delete(key: string): Promise<boolean>;
}

export interface OptimizeState {
  initialized: boolean;
  cache: ICache | null;
  profile: OptimizeProfile | null;
}

export const _state: OptimizeState = {
  initialized: false,
  cache: null,
  profile: null,
};
