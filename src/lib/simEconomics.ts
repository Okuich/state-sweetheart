// Simulation Economics Engine
// ─────────────────────────────────────────────────────────────────────
// Closed-form cost model for the live SimParams. We don't actually meter
// the GPU; we estimate FLOPs/step from N, algorithm, integrator, and
// substeps, then translate into wall-time, energy (Joules), CO₂ grams
// and $-cost using device-class coefficients. This is deliberately a
// *model* — calibrated against rough back-of-envelope numbers for a
// laptop CPU vs an entry GPU — so users can compare regimes without
// running expensive benchmarks.

import type { SimParams } from "@/components/PhysicsCanvas";

export type DeviceClass = "cpu_laptop" | "cpu_server" | "gpu_consumer" | "gpu_datacenter";

export type DeviceProfile = {
  id: DeviceClass;
  label: string;
  // Effective sustained throughput (GFLOP/s) on this kind of workload —
  // not peak; reflects memory-bound n-body / spring kernels.
  gflops: number;
  // Idle + active power envelope (Watts) and $/hour amortized.
  tdpW: number;
  idleW: number;
  costPerHour: number;
  // Grid intensity gCO₂/kWh — averaged over typical region for the class.
  gridIntensity: number;
};

export const DEVICE_PROFILES: Record<DeviceClass, DeviceProfile> = {
  cpu_laptop:    { id: "cpu_laptop",    label: "CPU · laptop (M-class)",     gflops:  60, tdpW:  28, idleW:  6, costPerHour: 0.02, gridIntensity: 380 },
  cpu_server:    { id: "cpu_server",    label: "CPU · server (32-core)",     gflops: 450, tdpW: 220, idleW: 60, costPerHour: 0.34, gridIntensity: 320 },
  gpu_consumer:  { id: "gpu_consumer",  label: "GPU · consumer (RTX-class)", gflops:8000, tdpW: 320, idleW: 25, costPerHour: 0.45, gridIntensity: 380 },
  gpu_datacenter:{ id: "gpu_datacenter",label: "GPU · datacenter (A/H-class)",gflops:24000,tdpW: 400, idleW: 50, costPerHour: 2.10, gridIntensity: 240 },
};

// FLOP estimate for one *physics step* (before substeps multiplier).
// We split contributions because the report shows where the budget goes.
export type CostBreakdown = {
  flopsIntegrate: number;
  flopsPairwise: number;
  flopsSprings: number;
  flopsField: number;
  flopsConstraints: number;
  flopsEnsemble: number;
  flopsTotal: number;          // already × substeps
};

const INTEGRATOR_COST: Record<SimParams["integrator"], number> = {
  "euler": 12, "semi-euler": 14, "verlet": 22,
};

export function estimateStepFlops(p: SimParams): CostBreakdown {
  const N = p.particleCount;
  const sub = Math.max(1, p.subSteps);

  // Integrate: ~K ops per particle per substep.
  const flopsIntegrate = N * INTEGRATOR_COST[p.integrator];

  // Pairwise force: O(N·k̄) for grid (k̄ ≈ ρπr²), O(N²) for all-pairs.
  // Per interaction ~30 FLOPs (distance + force kernel).
  const cellArea = Math.PI * p.pairwiseRadius * p.pairwiseRadius;
  const sceneArea = 1280 * 720; // typical canvas
  const expectedNeighbors = Math.min(N - 1, Math.max(1, (N - 1) * cellArea / sceneArea));
  const pairOps = p.pairwiseAlgo === "grid"
    ? N * expectedNeighbors * 30
    : N * (N - 1) * 15; // /2 symmetry already baked in

  // Springs: edges × edgesPerNode.
  const flopsSprings = N * p.edgesPerNode * 18;

  // Field eval: ~25 ops/sample (grad). finite-diff doubles cost.
  const fieldMul = p.field === "none" ? 0 : (p.potentialGrad === "finite-diff" ? 50 : 25);
  const flopsField = N * fieldMul * p.fieldStrength > 0 ? N * fieldMul : 0;

  // Constraint relaxation: iters × edges × cheap ops.
  const flopsConstraints = N * p.edgesPerNode * p.constraintIters * 12;

  // Monte Carlo ensemble: K replicas advected linearly per particle.
  const flopsEnsemble = p.stochastic ? N * p.ensembleK * 10 : 0;

  const perSubstep = flopsIntegrate + pairOps + flopsSprings + flopsField + flopsConstraints + flopsEnsemble;

  return {
    flopsIntegrate: flopsIntegrate * sub,
    flopsPairwise:  pairOps * sub,
    flopsSprings:   flopsSprings * sub,
    flopsField:     flopsField * sub,
    flopsConstraints: flopsConstraints * sub,
    flopsEnsemble:  flopsEnsemble * sub,
    flopsTotal:     perSubstep * sub,
  };
}

// Translate FLOPs → time/energy/$ on a given device, assuming `fps`
// target frame rate.
export type CostReport = {
  device: DeviceProfile;
  breakdown: CostBreakdown;
  msPerStep: number;
  utilization: number;     // 0..1 of the device's headroom at this fps
  watts: number;           // active power draw
  joulesPerStep: number;
  joulesPerHour: number;
  costPerHour: number;     // $
  gCO2PerHour: number;
  // Headroom signal: > 1 means we can't sustain fps on this device.
  saturation: number;
  // f32 vs f64 penalty already folded into utilization for CPU; on GPU
  // f64 is a hard 8× penalty (WGSL emulation).
  precisionPenalty: number;
};

export function estimateCost(p: SimParams, device: DeviceClass, targetFps = 60): CostReport {
  const profile = DEVICE_PROFILES[device];
  const breakdown = estimateStepFlops(p);

  // Precision penalty
  const isGpu = device.startsWith("gpu");
  const precisionPenalty = p.dtype === "float64" ? (isGpu ? 8 : 2) : 1;

  // Effective throughput
  const effGflops = profile.gflops / precisionPenalty;
  const flopsPerSec = effGflops * 1e9;

  const secondsPerStep = breakdown.flopsTotal / flopsPerSec;
  const msPerStep = secondsPerStep * 1000;

  const stepsPerSec = targetFps;
  const utilization = Math.min(1, secondsPerStep * stepsPerSec);
  const saturation = secondsPerStep * stepsPerSec; // unclamped
  const watts = profile.idleW + (profile.tdpW - profile.idleW) * utilization;

  const joulesPerStep = watts * secondsPerStep;
  const joulesPerHour = watts * 3600;
  const kWhPerHour = watts / 1000;

  return {
    device: profile,
    breakdown,
    msPerStep,
    utilization,
    watts,
    joulesPerStep,
    joulesPerHour,
    costPerHour: profile.costPerHour, // $ amortized — flat regardless of util
    gCO2PerHour: kWhPerHour * profile.gridIntensity,
    saturation,
    precisionPenalty,
  };
}

// Energy-aware scheduling: rank devices by gCO₂ for THIS workload sustained
// at target fps. Penalize devices that can't sustain (saturation > 1) by
// scaling the energy linearly with the slowdown.
export type ScheduleOption = {
  device: DeviceClass;
  label: string;
  feasible: boolean;
  effectiveGCO2PerHour: number;
  effectiveCostPerHour: number;
  msPerStep: number;
  saturation: number;
  rationale: string;
};

export function rankDevices(p: SimParams, targetFps = 60): ScheduleOption[] {
  const opts: ScheduleOption[] = (Object.keys(DEVICE_PROFILES) as DeviceClass[]).map((id) => {
    const r = estimateCost(p, id, targetFps);
    const feasible = r.saturation <= 1;
    const slowdown = Math.max(1, r.saturation);
    const eff = r.gCO2PerHour * slowdown;
    const reasons: string[] = [];
    reasons.push(`${r.msPerStep.toFixed(2)} ms/step`);
    reasons.push(`util ${(r.utilization * 100).toFixed(0)}%`);
    if (!feasible) reasons.push(`⚠ ${slowdown.toFixed(1)}× too slow for ${targetFps} fps`);
    if (r.precisionPenalty > 1) reasons.push(`f64 penalty ×${r.precisionPenalty}`);
    return {
      device: id,
      label: r.device.label,
      feasible,
      effectiveGCO2PerHour: eff,
      effectiveCostPerHour: r.costPerHour * slowdown,
      msPerStep: r.msPerStep,
      saturation: r.saturation,
      rationale: reasons.join(" · "),
    };
  });
  // Greenest feasible first; infeasible at end.
  opts.sort((a, b) => {
    if (a.feasible !== b.feasible) return a.feasible ? -1 : 1;
    return a.effectiveGCO2PerHour - b.effectiveGCO2PerHour;
  });
  return opts;
}

// GPU efficiency / cluster recommendations: structured tips tied to
// concrete SimParams patches the user can apply.
export type EfficiencyTip = {
  id: string;
  title: string;
  detail: string;
  estSavingPct: number;     // estimated FLOP reduction
  patch?: Partial<SimParams>;
};

export function efficiencyTips(p: SimParams): EfficiencyTip[] {
  const tips: EfficiencyTip[] = [];
  const bd = estimateStepFlops(p);

  if (p.pairwiseAlgo === "all-pairs" && p.particleCount > 300) {
    const ratio = bd.flopsPairwise / bd.flopsTotal;
    tips.push({
      id: "grid",
      title: "Switch pairwise to spatial grid",
      detail: `O(N²) is ${(ratio * 100).toFixed(0)}% of the budget. Grid drops it to O(N·k̄).`,
      estSavingPct: ratio * 90,
      patch: { pairwiseAlgo: "grid" },
    });
  }
  if (p.dtype === "float64") {
    tips.push({
      id: "f32",
      title: "Drop to float32",
      detail: "f64 doubles memory traffic on CPU and is ~8× slower on GPU (WGSL emulation).",
      estSavingPct: 50,
      patch: { dtype: "float32" },
    });
  }
  if (!p.adaptiveSubSteps && p.subSteps > 2) {
    tips.push({
      id: "adapt",
      title: "Enable adaptive substeps",
      detail: "Static high subSteps wastes compute when the system is calm.",
      estSavingPct: 25,
      patch: { adaptiveSubSteps: true },
    });
  }
  if (p.potentialGrad === "finite-diff" && p.field !== "none") {
    tips.push({
      id: "analytic",
      title: "Use analytic gradient for the field",
      detail: "Finite difference doubles field-eval cost vs analytic ∇Φ.",
      estSavingPct: 8,
      patch: { potentialGrad: "analytic" },
    });
  }
  if (p.stochastic && p.ensembleK > 24) {
    tips.push({
      id: "mcK",
      title: `Cap ensemble at K=24 (currently ${p.ensembleK})`,
      detail: "σ estimates plateau quickly past K≈24. Higher K is mostly variance polishing.",
      estSavingPct: ((p.ensembleK - 24) / p.ensembleK) * (bd.flopsEnsemble / bd.flopsTotal) * 100,
      patch: { ensembleK: 24 },
    });
  }
  if (p.constraintIters > 6 && p.springK < 100) {
    tips.push({
      id: "iters",
      title: "Reduce constraint iterations",
      detail: "Soft springs converge in ≤4 relaxation passes; extra iters are wasted.",
      estSavingPct: 6,
      patch: { constraintIters: 4 },
    });
  }
  return tips.sort((a, b) => b.estSavingPct - a.estSavingPct);
}

export function fmtFlops(f: number): string {
  if (f > 1e12) return (f / 1e12).toFixed(2) + " TF";
  if (f > 1e9) return (f / 1e9).toFixed(2) + " GF";
  if (f > 1e6) return (f / 1e6).toFixed(2) + " MF";
  if (f > 1e3) return (f / 1e3).toFixed(2) + " kF";
  return f.toFixed(0) + " F";
}
