// Autonomous Physics Agents — agentic reasoning layer over SimParams.
//
// Architecture: each agent owns a *capability* (construct, optimize, diagnose,
// stabilize, recommend). Agents read the current world state (SimParams +
// optional ValidationReport + loss) and emit `AgentAction`s — typed proposals
// that the orchestrator can dispatch through a small tool registry.
//
// We deliberately avoid a heavy LLM dep here: this is a deterministic,
// auditable reasoning layer. Each rule encodes a piece of physics intuition
// ("if explicit Euler + high spring K → unstable; switch to verlet, raise
// substeps") and exposes its rationale so the UI can show *why* it fired.

import type { SimParams, ValidationReport } from "@/components/PhysicsCanvas";

export type AgentKind =
  | "constructor"   // builds / seeds a simulation from a goal
  | "optimizer"     // proposes parameter moves to reduce a loss
  | "diagnostician" // detects anomalies / pathological regimes
  | "stabilizer"    // corrective actions for blow-ups, NaNs, stretch
  | "designer";     // long-horizon design recommendations

export type AgentSeverity = "info" | "warn" | "critical";

// A patch is a partial SimParams diff the agent wants applied.
export type ParamPatch = Partial<SimParams>;

export type AgentAction = {
  id: string;
  agent: AgentKind;
  title: string;
  rationale: string;        // human-readable "why"
  severity: AgentSeverity;
  patch?: ParamPatch;       // if present, can be one-click applied
  tool?: ToolCall;          // alternative: invoke a tool instead of patching
  confidence: number;       // 0..1 — how strongly the agent endorses this
};

// ── Tool registry ──────────────────────────────────────────────────────
// "Tool usage" capability: agents emit tool calls instead of touching
// params directly. The orchestrator owns dispatch so the UI can preview &
// approve. Tools are pure data — execution lives in the host component.

export type ToolCall =
  | { name: "reset_sim" }
  | { name: "snapshot_world"; label: string }
  | { name: "apply_patch"; patch: ParamPatch }
  | { name: "set_preset"; preset: PresetName };

export type PresetName = "stable_lattice" | "chaotic_swarm" | "energy_study" | "stress_test";

export const PRESET_PATCHES: Record<PresetName, ParamPatch> = {
  stable_lattice: {
    integrator: "verlet", subSteps: 4, constraintIters: 6,
    damping: 0.6, springK: 60, pairwiseStrength: 80, adaptiveSubSteps: true,
  },
  chaotic_swarm: {
    integrator: "semi-euler", subSteps: 2, damping: 0.1,
    pairwiseStrength: 320, pairwiseRadius: 70, attractor: 2.4, field: "swirl",
  },
  energy_study: {
    integrator: "verlet", dtype: "float64", subSteps: 6,
    damping: 0, gravity: 0, optimize: false,
  },
  stress_test: {
    particleCount: 1200, pairwiseAlgo: "grid", integrator: "verlet",
    subSteps: 3, adaptiveSubSteps: true, maxSubSteps: 16,
  },
};

// ── Agent inputs ───────────────────────────────────────────────────────
export type AgentContext = {
  params: SimParams;
  validation: ValidationReport | null;
  loss: number | null;
  goal?: AgentGoal;
};

export type AgentGoal =
  | "stability"
  | "performance"
  | "exploration"
  | "energy_conservation"
  | "minimize_loss";

// ── Individual agent rule sets ─────────────────────────────────────────

let _idc = 0;
const nid = (k: string) => `${k}-${++_idc}-${Date.now().toString(36)}`;

function constructor_(ctx: AgentContext): AgentAction[] {
  const out: AgentAction[] = [];
  const { params, goal } = ctx;
  if (goal === "stability" && params.integrator !== "verlet") {
    out.push({
      id: nid("ctor"), agent: "constructor", severity: "info", confidence: 0.85,
      title: "Seed a stable lattice configuration",
      rationale:
        "Goal=stability. Verlet is symplectic and conserves energy better than (semi-)Euler. " +
        "Pre-baking damping=0.6 and constraintIters=6 prevents cold-start blow-ups.",
      tool: { name: "set_preset", preset: "stable_lattice" },
    });
  }
  if (goal === "exploration") {
    out.push({
      id: nid("ctor"), agent: "constructor", severity: "info", confidence: 0.7,
      title: "Construct chaotic swarm regime",
      rationale: "High pairwise strength + swirl field maximizes phase-space coverage for exploration.",
      tool: { name: "set_preset", preset: "chaotic_swarm" },
    });
  }
  if (goal === "energy_conservation") {
    out.push({
      id: nid("ctor"), agent: "constructor", severity: "info", confidence: 0.9,
      title: "Set up energy conservation study",
      rationale: "float64 + verlet + 6 substeps + zero damping — reference setup for drift measurement.",
      tool: { name: "set_preset", preset: "energy_study" },
    });
  }
  return out;
}

function diagnostician(ctx: AgentContext): AgentAction[] {
  const out: AgentAction[] = [];
  const { params, validation } = ctx;

  // NaN / shape issues from runtime validator → critical
  if (validation && !validation.ok) {
    out.push({
      id: nid("diag"), agent: "diagnostician", severity: "critical", confidence: 0.99,
      title: `Tensor invariant violated (${validation.issues.length} issue${validation.issues.length === 1 ? "" : "s"})`,
      rationale:
        "Runtime shape/dtype/finiteness check failed — likely NaN propagation from too-large dt × force. " +
        "First issues: " + validation.issues.slice(0, 3).map((i) => `${i.field}: expected ${i.expected}, got ${i.got}`).join(" · "),
    });
  }

  // CFL-ish heuristic: stiff springs with explicit Euler
  const stiffness = params.springK * Math.max(1, params.edgesPerNode);
  if (params.integrator === "euler" && stiffness > 200) {
    out.push({
      id: nid("diag"), agent: "diagnostician", severity: "warn", confidence: 0.88,
      title: "Stiff system on explicit Euler — instability likely",
      rationale:
        `springK·edgesPerNode = ${stiffness.toFixed(0)} ≫ stable bound for forward Euler. ` +
        "CFL: dt < 2/√(k/m). Switch to verlet or raise substeps.",
    });
  }

  // Pairwise neighbourhood explosion
  const neighborhood = params.particleCount * (params.pairwiseRadius / 60);
  if (params.pairwiseAlgo === "all-pairs" && params.particleCount > 500) {
    out.push({
      id: nid("diag"), agent: "diagnostician", severity: "warn", confidence: 0.92,
      title: "O(N²) pairwise on N>500 — perf cliff",
      rationale: `N=${params.particleCount}, all-pairs is ~${(params.particleCount ** 2 / 1e6).toFixed(1)}M ops/step. Switch to grid.`,
      patch: { pairwiseAlgo: "grid" },
    });
  } else if (neighborhood > 800) {
    out.push({
      id: nid("diag"), agent: "diagnostician", severity: "info", confidence: 0.6,
      title: "Dense neighborhood — consider tightening pairwise radius",
      rationale: `Effective neighborhood ≈ ${neighborhood.toFixed(0)}. Smaller radius = better grid-cell occupancy.`,
    });
  }

  // Field sampling out of bounds for periodic boundary
  if (params.boundary === "periodic" && params.fieldSampling === "clamp") {
    out.push({
      id: nid("diag"), agent: "diagnostician", severity: "info", confidence: 0.7,
      title: "Field/boundary sampling mismatch",
      rationale: "Periodic boundary with clamp sampling creates discontinuities at wrap edges. Use 'wrap' sampling.",
      patch: { fieldSampling: "wrap" },
    });
  }

  return out;
}

function stabilizer(ctx: AgentContext): AgentAction[] {
  const out: AgentAction[] = [];
  const { params, validation } = ctx;

  if (validation && !validation.ok) {
    out.push({
      id: nid("stab"), agent: "stabilizer", severity: "critical", confidence: 0.95,
      title: "Emergency: hard-reset and lower step",
      rationale: "Tensor check failed. Halving substep budget is rarely enough — full reset + verlet is the safe path.",
      tool: { name: "reset_sim" },
    });
    out.push({
      id: nid("stab"), agent: "stabilizer", severity: "critical", confidence: 0.9,
      title: "Switch to verlet + raise substeps to 4",
      rationale: "Symplectic integrator + smaller effective dt restores stability for stiff regimes.",
      patch: { integrator: "verlet", subSteps: Math.max(4, params.subSteps), adaptiveSubSteps: true },
    });
  }

  if (params.gravity > 200 && params.damping < 0.1) {
    out.push({
      id: nid("stab"), agent: "stabilizer", severity: "warn", confidence: 0.7,
      title: "Add damping to bleed kinetic energy",
      rationale: "High gravity with no damping leads to runaway impacts at the boundary.",
      patch: { damping: 0.25 },
    });
  }

  if (params.springK > 150 && params.constraintIters < 4) {
    out.push({
      id: nid("stab"), agent: "stabilizer", severity: "warn", confidence: 0.75,
      title: "Increase constraint iterations",
      rationale: "Stiff springs need more relaxation passes to stay near rest length each step.",
      patch: { constraintIters: 6 },
    });
  }

  return out;
}

function optimizer(ctx: AgentContext): AgentAction[] {
  const out: AgentAction[] = [];
  const { params, loss, goal } = ctx;

  if (goal === "performance") {
    if (params.particleCount > 600 && params.pairwiseAlgo !== "grid") {
      out.push({
        id: nid("opt"), agent: "optimizer", severity: "info", confidence: 0.95,
        title: "Switch pairwise force to spatial grid",
        rationale: "O(N) with grid vs O(N²) all-pairs — ~50× speedup at N=800.",
        patch: { pairwiseAlgo: "grid" },
      });
    }
    if (params.dtype === "float64") {
      out.push({
        id: nid("opt"), agent: "optimizer", severity: "info", confidence: 0.8,
        title: "Drop to float32 for ~2× throughput",
        rationale: "f64 doubles memory traffic and halves SIMD lanes. Use f32 unless precision-bound.",
        patch: { dtype: "float32" },
      });
    }
    if (params.subSteps > 4 && !params.adaptiveSubSteps) {
      out.push({
        id: nid("opt"), agent: "optimizer", severity: "info", confidence: 0.7,
        title: "Enable adaptive substeps to amortize cost",
        rationale: "Static high subSteps wastes compute on quiet frames. Adaptive grows only when needed.",
        patch: { adaptiveSubSteps: true },
      });
    }
  }

  if (goal === "minimize_loss" || params.optimize) {
    if (loss != null && Math.abs(loss) > 10 && params.objectiveLR < 0.1) {
      out.push({
        id: nid("opt"), agent: "optimizer", severity: "info", confidence: 0.6,
        title: "Raise learning rate (loss large, lr small)",
        rationale: `|loss|=${Math.abs(loss).toExponential(2)} suggests under-stepping at lr=${params.objectiveLR}.`,
        patch: { objectiveLR: Math.min(0.5, params.objectiveLR * 2) },
      });
    }
    if (loss != null && Math.abs(loss) < 1e-3 && params.objectiveLR > 0.02) {
      out.push({
        id: nid("opt"), agent: "optimizer", severity: "info", confidence: 0.7,
        title: "Anneal learning rate (near optimum)",
        rationale: `|loss|=${Math.abs(loss).toExponential(2)} is small — shrink lr to refine.`,
        patch: { objectiveLR: params.objectiveLR * 0.5 },
      });
    }
    if (!params.optimize) {
      out.push({
        id: nid("opt"), agent: "optimizer", severity: "info", confidence: 0.5,
        title: "Engage objective optimizer",
        rationale: "Goal=minimize_loss but optimize flag is off.",
        patch: { optimize: true },
      });
    }
  }
  return out;
}

function designer(ctx: AgentContext): AgentAction[] {
  const out: AgentAction[] = [];
  const { params } = ctx;

  if (!params.stochastic && (params.integrator === "verlet" || params.dtype === "float64")) {
    out.push({
      id: nid("des"), agent: "designer", severity: "info", confidence: 0.55,
      title: "Add Monte Carlo ensemble for uncertainty bands",
      rationale: "You're paying for high-fidelity integration — pair it with K=12 replicas to quantify σ_x(t).",
      patch: { stochastic: true, ensembleK: Math.max(12, params.ensembleK) },
    });
  }

  if (params.particleCount < 200 && params.pairwiseStrength > 100) {
    out.push({
      id: nid("des"), agent: "designer", severity: "info", confidence: 0.5,
      title: "Scale up N to populate the configuration space",
      rationale: "Strong pairwise forces with low N produces sparse phase portraits — try N≈600.",
      patch: { particleCount: 600 },
    });
  }

  if (params.field === "none" && params.attractor < 0.5) {
    out.push({
      id: nid("des"), agent: "designer", severity: "info", confidence: 0.45,
      title: "Add a potential field for richer dynamics",
      rationale: "No external field + weak attractor → diffusive uninteresting trajectories.",
      patch: { field: "swirl", fieldStrength: 0.6 },
    });
  }

  // Snapshot recommendation: if nothing recently saved & loss is converging
  out.push({
    id: nid("des"), agent: "designer", severity: "info", confidence: 0.4,
    title: "Snapshot current world to long-term memory",
    rationale: "Recommended after non-trivial parameter exploration so future runs can retrieve & restore.",
    tool: { name: "snapshot_world", label: `auto-${new Date().toISOString().slice(11, 19)}` },
  });

  return out;
}

// ── Orchestrator ───────────────────────────────────────────────────────
// Distributed orchestration model: each agent runs independently and
// emits actions in parallel. The orchestrator merges, dedupes by patch
// fingerprint, and sorts by (severity, confidence).

const SEV_RANK: Record<AgentSeverity, number> = { critical: 3, warn: 2, info: 1 };

function fingerprint(a: AgentAction): string {
  if (a.patch) return "p:" + JSON.stringify(a.patch);
  if (a.tool) return "t:" + JSON.stringify(a.tool);
  return "x:" + a.title;
}

export function runAgents(ctx: AgentContext, enabled?: Partial<Record<AgentKind, boolean>>): AgentAction[] {
  const e = { constructor: true, optimizer: true, diagnostician: true, stabilizer: true, designer: true, ...(enabled ?? {}) };
  const all: AgentAction[] = [];
  if (e.constructor) all.push(...constructor_(ctx));
  if (e.diagnostician) all.push(...diagnostician(ctx));
  if (e.stabilizer) all.push(...stabilizer(ctx));
  if (e.optimizer) all.push(...optimizer(ctx));
  if (e.designer) all.push(...designer(ctx));

  const seen = new Set<string>();
  const deduped: AgentAction[] = [];
  for (const a of all) {
    const fp = fingerprint(a);
    if (seen.has(fp)) continue;
    seen.add(fp);
    deduped.push(a);
  }
  deduped.sort((x, y) => {
    const dr = SEV_RANK[y.severity] - SEV_RANK[x.severity];
    if (dr !== 0) return dr;
    return y.confidence - x.confidence;
  });
  return deduped;
}

// Semantic reasoning helper: terse natural-language summary of world state.
export function summarizeWorld(p: SimParams): string {
  const bits: string[] = [];
  bits.push(`${p.particleCount} particles · ${p.integrator}/${p.dtype}`);
  bits.push(`pairwise=${p.pairwiseAlgo}(${p.pairwiseMode}) k=${p.springK} γ=${p.damping}`);
  if (p.field !== "none") bits.push(`field=${p.field}×${p.fieldStrength.toFixed(2)}`);
  if (p.optimize) bits.push(`optimize lr=${p.objectiveLR}`);
  if (p.stochastic) bits.push(`MC K=${p.ensembleK} σ=${p.noiseSigma}`);
  if (p.twinEnabled) bits.push(`twin M=${p.twinSensorCount} g=${p.twinAssimGain}`);
  return bits.join(" · ");
}
