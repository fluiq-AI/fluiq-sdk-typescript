/**
 * Prompt template fetched from the Fluiq dashboard.
 *
 * Mirrors fluiq.prompts.Prompt from the Python SDK.
 */

export interface PromptData {
  slug: string;
  name?: string;
  template: string;
  model?: string | null;
  variables?: string[];
  version?: number;
  environment?: string;
}

export class Prompt {
  /** The prompt's unique identifier. */
  readonly slug: string;
  /** Human-readable display name. */
  readonly name: string;
  /** Raw template string with `{variable}` placeholders. */
  readonly template: string;
  /** Preferred model pinned in the dashboard, if any. */
  readonly model: string | null;
  /** Declared variable names. */
  readonly variables: string[];
  /** Deployed version number. */
  readonly version: number;
  /** Environment this snapshot was fetched from. */
  readonly environment: string;

  constructor(data: PromptData) {
    this.slug = data.slug;
    this.name = data.name ?? "";
    this.template = data.template;
    this.model = data.model ?? null;
    this.variables = data.variables ?? [];
    this.version = data.version ?? 1;
    this.environment = data.environment ?? "production";
  }

  /**
   * Render the template by substituting `{variable}` placeholders.
   *
   * Mirrors Python's `str.format(**kwargs)`: `{{` / `}}` are literal braces and
   * a missing variable throws (the TS analogue of Python's KeyError).
   */
  render(variables: Record<string, string | number> = {}): string {
    return this.template.replace(
      /\{\{|\}\}|\{([^{}]*)\}/g,
      (match, name?: string) => {
        if (match === "{{") return "{";
        if (match === "}}") return "}";
        const key = (name ?? "").trim();
        if (!(key in variables)) {
          throw new Error(`Missing template variable: '${key}'`);
        }
        return String(variables[key]);
      }
    );
  }

  toString(): string {
    return `<Prompt slug='${this.slug}' version=${this.version} env='${this.environment}'>`;
  }
}
