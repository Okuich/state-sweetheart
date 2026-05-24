// Autonomous Engineering Intelligence — types
export interface DesignVars {
  // parametric bracket: a beam with two mount holes and a web
  length: number;       // mm
  width: number;        // mm
  thickness: number;    // mm
  filletR: number;      // mm
  holeR: number;        // mm
  webRatio: number;     // 0..1 thickness of internal web
  material: "Al6061" | "Steel1018" | "Ti6Al4V" | "PA12-CF";
}

export interface Objectives {
  mass: number;            // kg (minimize)
  stressMax: number;       // MPa (constraint: < yield * safety)
  deflection: number;      // mm (minimize)
  cost: number;            // USD (minimize)
  energyMJ: number;        // embodied + machining (minimize)
  fabMinutes: number;      // (minimize)
}

export interface FabConstraints {
  minWall: number;         // mm
  minHoleR: number;        // mm
  maxAspect: number;       // length / thickness
  safetyFactor: number;
  maxStressFraction: number; // of yield
}

export interface Candidate {
  id: string;
  vars: DesignVars;
  obj: Objectives;
  feasible: boolean;
  violations: string[];
  score: number;           // composite (lower = better)
  rank: number;            // Pareto front rank
  crowding: number;
  generation: number;
}

export interface RunMetrics {
  generations: number;
  evaluations: number;
  paretoSize: number;
  hypervolume: number;
  feasibleRatio: number;
  workflowAccelPct: number;  // vs random search baseline
  fabOptGainPct: number;     // best fab vs population mean
  prototypeReductionPct: number; // candidates avoided by surrogate filter
  bestKnownScore: number;
  elapsedMs: number;
}

export type MaterialKey = DesignVars["material"];
