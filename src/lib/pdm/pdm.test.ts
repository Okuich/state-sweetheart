import { describe, it, expect, beforeEach } from "vitest";
import {
  PredictiveMaintenanceEngine, CHANNEL_ORDER, resetSeed,
  packReading, fitEmbedding, project,
  type SensorReading,
} from "./index";

function r(assetId: string, t: number, base: number, drift = 0): SensorReading {
  return {
    assetId, t,
    channels: {
      vibration:   1 + base * 0.1 + drift * 0.4,
      pressure:    10 + base * 0.05 + drift * 0.3,
      temperature: 60 + base * 0.2 + drift * 3.0,
      flow_rate:   5  - drift * 0.2,
      current:     12 + base * 0.05 + drift * 0.6,
      acoustic:    0.3 + drift * 0.3,
      hydraulic:   2.0 + drift * 0.2,
    },
  };
}

describe("PDM embeddings", () => {
  beforeEach(() => resetSeed(123));

  it("packReading produces a dense vector of CHANNEL_ORDER length", () => {
    const v = packReading({ assetId: "x", t: 0, channels: { vibration: 1.5, temperature: 70 } });
    expect(v.length).toBe(CHANNEL_ORDER.length);
    expect(v[CHANNEL_ORDER.indexOf("vibration")]).toBe(1.5);
    expect(v[CHANNEL_ORDER.indexOf("temperature")]).toBe(70);
  });

  it("PCA basis is approximately orthonormal in the top-k subspace", () => {
    const samples = Array.from({ length: 32 }, (_, i) => packReading(r("a", i, i)));
    const m = fitEmbedding(samples, 3);
    for (let i = 0; i < m.k; i++) {
      let n = 0;
      for (let j = 0; j < m.d; j++) n += m.basis[i * m.d + j] ** 2;
      expect(Math.sqrt(n)).toBeCloseTo(1, 5);
    }
    // First component captures dominant variance > others.
    expect(m.sigma[0]).toBeGreaterThanOrEqual(m.sigma[m.sigma.length - 1]);
  });

  it("project is linear: project(a+b) = project(a)+project(b) (centered)", () => {
    const samples = Array.from({ length: 24 }, (_, i) => packReading(r("a", i, i)));
    const m = fitEmbedding(samples, 2);
    const a = samples[3], b = samples[7];
    const pa = project(m, a), pb = project(m, b);
    const sum = new Float64Array(a.length);
    // Linearity holds for centered+scaled inputs; we test the deterministic
    // composition path through project() vs. the analytic sum.
    for (let i = 0; i < a.length; i++) sum[i] = a[i] + b[i] - m.mean[i];
    const ps = project(m, sum);
    for (let i = 0; i < m.k; i++) expect(ps[i]).toBeCloseTo(pa[i] + pb[i], 6);
  });
});

describe("PredictiveMaintenanceEngine", () => {
  beforeEach(() => resetSeed(123));

  it("scores healthy steady-state as ok and rising drift escalates priority", () => {
    const eng = new PredictiveMaintenanceEngine({ latentDim: 3, warmupSamples: 20 });
    // Warm up with healthy data.
    for (let i = 0; i < 60; i++) eng.ingest(r("pump-1", i, Math.sin(i / 5)));
    const ok = eng.snapshot("pump-1")!;
    expect(ok.priority === "ok" || ok.priority === "watch").toBe(true);
    expect(ok.riskScore).toBeLessThan(0.5);

    // Inject worsening drift.
    let last = ok;
    for (let i = 60; i < 90; i++) last = eng.ingest(r("pump-1", i, Math.sin(i / 5), (i - 60) * 0.4));
    expect(last.riskScore).toBeGreaterThan(ok.riskScore);
    expect(["watch", "schedule", "urgent"]).toContain(last.priority);
    expect(last.driftSigma).toBeGreaterThan(ok.driftSigma);
  });

  it("exemplar-aware scoring penalizes states near registered failures", () => {
    const eng = new PredictiveMaintenanceEngine({ latentDim: 3, warmupSamples: 10 });
    const optimal = Array.from({ length: 12 }, (_, i) => r("v1", i, Math.sin(i)));
    const failures = Array.from({ length: 8 }, (_, i) => r("v1", 1000 + i, 0, 6));
    eng.registerExemplars("v1", { optimal, failures });
    for (let i = 0; i < 30; i++) eng.ingest(r("v1", i, Math.sin(i)));
    const healthy = eng.snapshot("v1")!;
    const bad = eng.ingest(r("v1", 31, 0, 6));
    expect(bad.distToFailure).toBeLessThan(healthy.distToFailure || Infinity);
    expect(bad.riskScore).toBeGreaterThan(healthy.riskScore);
  });

  it("prioritized() returns assets in descending urgency", () => {
    const eng = new PredictiveMaintenanceEngine({ latentDim: 3, warmupSamples: 10 });
    for (let i = 0; i < 30; i++) {
      eng.ingest(r("calm", i, Math.sin(i / 4)));
      eng.ingest(r("loud", i, Math.sin(i / 4), i * 0.25));
    }
    const list = eng.prioritized();
    expect(list.length).toBe(2);
    expect(list[0].assetId).toBe("loud");
    expect(list[0].riskScore).toBeGreaterThanOrEqual(list[1].riskScore);
  });
});
