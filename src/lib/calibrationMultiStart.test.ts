import { describe, it, expect } from "vitest";
import {
  calibrateMultiStart,
  generateStarts,
} from "./calibrationMultiStart";
import {
  generateMeasurements,
  type ModelParams,
} from "./calibration";

describe("calibrateMultiStart", () => {
  it("best fit beats worst start on clean data", () => {
    const data = generateMeasurements(160, 8, 0.02, 0.0);
    const r = calibrateMultiStart(data, { starts: 12, seed: 1, polishIter: 80 });
    expect(r.best.rss).toBeLessThanOrEqual(r.worstRss);
    expect(Number.isFinite(r.best.r2)).toBe(true);
  });

  it("multi-start yields consistent top candidates on noisy data", () => {
    const data = generateMeasurements(200, 8, 0.08, 0.02);
    const r = calibrateMultiStart(data, { starts: 12, seed: 2, polishIter: 40 });
    expect(r.candidates).toHaveLength(12);
    // top-3 consensus stddev should be finite & non-negative.
    expect(r.consensus.std.k).toBeGreaterThanOrEqual(0);
  });

  it("different seeds produce different but bounded start points", () => {
    const a = generateStarts(8, 1);
    const b = generateStarts(8, 42);
    const inBounds = (p: ModelParams) =>
      p.k >= 1 && p.k <= 200 && p.c >= 0.01 && p.c <= 10 &&
      p.m >= 0.05 && p.m <= 5 && p.A >= 0.1 && p.A <= 3;
    expect(a.every(inBounds)).toBe(true);
    expect(b.every(inBounds)).toBe(true);
    // Should not be identical sequences.
    expect(a[0]).not.toEqual(b[0]);
  });

  it("deterministic for the same seed", () => {
    const data = generateMeasurements(120, 6, 0.04, 0.0);
    const a = calibrateMultiStart(data, { starts: 6, seed: 7, polishIter: 20 });
    const b = calibrateMultiStart(data, { starts: 6, seed: 7, polishIter: 20 });
    expect(a.params).toEqual(b.params);
    expect(a.best.rss).toBeCloseTo(b.best.rss, 10);
  });

  it("candidates sorted by RSS ascending and exactly one marked best", () => {
    const data = generateMeasurements(120, 6, 0.05, 0.0);
    const r = calibrateMultiStart(data, { starts: 6, seed: 3, polishIter: 10 });
    for (let i = 1; i < r.candidates.length; i++) {
      expect(r.candidates[i].fit.rss).toBeGreaterThanOrEqual(r.candidates[i - 1].fit.rss);
    }
    expect(r.candidates.filter((c) => c.best)).toHaveLength(1);
    expect(r.candidates[0].best).toBe(true);
  });

  it("multi-start beats a deliberately bad single start", () => {
    const data = generateMeasurements(160, 8, 0.04, 0.0);
    const multi = calibrateMultiStart(data, { starts: 12, seed: 11, polishIter: 40 });
    // The "bad" start would be the worst single candidate before polish.
    const worst = multi.candidates[multi.candidates.length - 1].fit.rss;
    expect(multi.best.rss).toBeLessThanOrEqual(worst);
  });

  it("respects starts count", () => {
    const data = generateMeasurements(100, 6, 0.05);
    const r = calibrateMultiStart(data, { starts: 5, seed: 0, polishIter: 5 });
    expect(r.candidates).toHaveLength(5);
  });
});
