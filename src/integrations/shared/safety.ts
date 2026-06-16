import { FluiqEvalError, FluiqSecurityError } from "../../exceptions";

/**
 * Wraps an async function so that any exception it throws is swallowed,
 * except for FluiqEvalError and FluiqSecurityError which are re-raised.
 * The SDK's observability pipeline must never crash the user's application.
 */
export function failOpen<T extends unknown[], R>(
  fn: (...args: T) => Promise<R>
): (...args: T) => Promise<R | undefined> {
  return async (...args: T): Promise<R | undefined> => {
    try {
      return await fn(...args);
    } catch (exc) {
      if (exc instanceof FluiqEvalError || exc instanceof FluiqSecurityError) throw exc;
      return undefined;
    }
  };
}

/**
 * Synchronous variant of failOpen.
 */
export function failOpenSync<T extends unknown[], R>(
  fn: (...args: T) => R
): (...args: T) => R | undefined {
  return (...args: T): R | undefined => {
    try {
      return fn(...args);
    } catch (exc) {
      if (exc instanceof FluiqEvalError || exc instanceof FluiqSecurityError) throw exc;
      return undefined;
    }
  };
}
