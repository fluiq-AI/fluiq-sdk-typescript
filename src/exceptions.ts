export class FluiqEvalError extends Error {
  failures: Record<string, number>;
  scores: Record<string, number>;

  constructor(failures: Record<string, number>, scores?: Record<string, number>) {
    const failedStr = Object.entries(failures)
      .map(([m, s]) => `${m}=${s.toFixed(3)}`)
      .join(", ");
    super(`Evaluation thresholds not met: ${failedStr}`);
    this.name = "FluiqEvalError";
    this.failures = failures;
    this.scores = scores ?? {};
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

export class FluiqSecurityError extends Error {
  blockReason: string;
  riskLevel: string;
  attackTypes: string[];

  constructor(blockReason: string, riskLevel = "high", attackTypes: string[] = []) {
    super(blockReason);
    this.name = "FluiqSecurityError";
    this.blockReason = blockReason;
    this.riskLevel = riskLevel;
    this.attackTypes = attackTypes;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}
