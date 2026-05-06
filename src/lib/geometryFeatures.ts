// Geometry Feature Intelligence Layer
// ─────────────────────────────────────────────────────────────
// Consumes a parsed STEP report + topology graph and emits:
//   - a topology descriptor (degree dist, components, depth)
//   - manufacturability features (thin walls, small holes, sharp
//     corners, deep pockets, undercuts) — heuristic from surface
//     mix + bbox aspect
//   - curvature signature (planar / cyl / cone / torus / spline mix)
//   - stress + thermal risk heuristics (sharp corners, thin walls)
//   - a fixed-length geometry embedding (32-dim) + fab feature vector
//
// All pure TS, derived from the canonical geometry — no external CAD math.

import type {
  ParseReport, TopoGraph, GeomDescriptor,
} from "@/lib/stepParser";

export type Severity = "low" | "medium" | "high";

export interface FeatureFinding {
  kind: string;
  count: number;
  severity: Severity;
  detail: string;
}

export interface TopologyDescriptor {
  components: number;
  maxDepth: number;
  meanOutDegree: number;
  maxOutDegree: number;
  faceShellRatio: number;
  edgeFaceRatio: number;
  euler: number;        // V - E + F (proxy)
  cyclomatic: number;   // E - V + components
}

export interface CurvatureSignature {
  planar: number;       // fraction
  cylindrical: number;
  conical: number;
  spherical: number;
  toroidal: number;
  spline: number;
  total: number;
  meanCurvatureProxy: number;
  gaussianCurvatureProxy: number;
}

export interface RiskHeuristics {
  stressConcentration: number;  // 0..1
  thermalRisk: number;          // 0..1
  fabricationDifficulty: number;
  notes: string[];
}

export interface FeatureIntelligence {
  topology: TopologyDescriptor;
  curvature: CurvatureSignature;
  findings: FeatureFinding[];
  risk: RiskHeuristics;
  fabFeatureVector: number[];   // 12-dim, normalized
  embedding: number[];          // 32-dim, normalized
  ms: number;
}

// Connected components on the undirected entity graph
function components(topo: TopoGraph): number {
  const seen = new Set<number>();
  let comps = 0;
  for (const id of topo.nodes.keys()) {
    if (seen.has(id)) continue;
    comps++;
    const stack = [id];
    while (stack.length) {
      const n = stack.pop()!;
      if (seen.has(n)) continue;
      seen.add(n);
      const out = topo.outEdges.get(n) ?? [];
      const inn = topo.inEdges.get(n) ?? [];
      for (const r of out) if (!seen.has(r) && topo.nodes.has(r)) stack.push(r);
      for (const r of inn) if (!seen.has(r) && topo.nodes.has(r)) stack.push(r);
    }
  }
  return comps;
}

// BFS depth from roots (capped)
function maxDepth(topo: TopoGraph): number {
  let max = 0;
  for (const r of topo.roots.slice(0, 32)) {
    const dist = new Map<number, number>([[r, 0]]);
    const q = [r];
    while (q.length) {
      const n = q.shift()!;
      const d = dist.get(n)!;
      if (d > max) max = d;
      if (d > 64) break;
      for (const c of topo.outEdges.get(n) ?? []) {
        if (!dist.has(c) && topo.nodes.has(c)) {
          dist.set(c, d + 1);
          q.push(c);
        }
      }
    }
  }
  return max;
}

function topologyDescriptor(report: ParseReport, topo: TopoGraph, desc: GeomDescriptor): TopologyDescriptor {
  let totalOut = 0, maxOut = 0;
  for (const refs of topo.outEdges.values()) {
    totalOut += refs.length;
    if (refs.length > maxOut) maxOut = refs.length;
  }
  const N = Math.max(topo.nodes.size, 1);
  const comps = components(topo);
  const V = desc.counts.vertices;
  const E = desc.counts.edges;
  const F = desc.counts.faces;
  return {
    components: comps,
    maxDepth: maxDepth(topo),
    meanOutDegree: totalOut / N,
    maxOutDegree: maxOut,
    faceShellRatio: F / Math.max(desc.counts.shells, 1),
    edgeFaceRatio: E / Math.max(F, 1),
    euler: V - E + F,
    cyclomatic: Math.max(0, E - V + comps),
  };
}

function curvatureSignature(desc: GeomDescriptor, report: ParseReport): CurvatureSignature {
  const C = (k: string) => report.byType.get(k)?.length ?? 0;
  const planar = C("PLANE");
  const cyl = C("CYLINDRICAL_SURFACE");
  const con = C("CONICAL_SURFACE");
  const sph = C("SPHERICAL_SURFACE");
  const tor = C("TOROIDAL_SURFACE");
  const spl = C("B_SPLINE_SURFACE_WITH_KNOTS") + C("B_SPLINE_SURFACE")
            + C("BEZIER_SURFACE") + C("RATIONAL_B_SPLINE_SURFACE");
  const total = planar + cyl + con + sph + tor + spl;
  const denom = Math.max(total, 1);
  // Mean / Gaussian curvature proxies (qualitative weights)
  const H = (cyl * 0.5 + con * 0.6 + sph * 1.0 + tor * 0.8 + spl * 0.9) / denom;
  const K = (sph * 1.0 + tor * 0.5 + spl * 0.7 - planar * 0.0) / denom;
  return {
    planar:      planar / denom,
    cylindrical: cyl / denom,
    conical:     con / denom,
    spherical:   sph / denom,
    toroidal:    tor / denom,
    spline:      spl / denom,
    total,
    meanCurvatureProxy: H,
    gaussianCurvatureProxy: K,
  };
}

function bboxDims(desc: GeomDescriptor): [number, number, number] {
  if (!desc.bbox) return [0, 0, 0];
  return [
    desc.bbox.max[0] - desc.bbox.min[0],
    desc.bbox.max[1] - desc.bbox.min[1],
    desc.bbox.max[2] - desc.bbox.min[2],
  ];
}

function findings(desc: GeomDescriptor, topo: TopologyDescriptor, curv: CurvatureSignature): FeatureFinding[] {
  const out: FeatureFinding[] = [];
  const dims = bboxDims(desc);
  const dmin = Math.min(...dims.filter((d) => d > 0), Infinity);
  const dmax = Math.max(...dims, 1e-9);
  const aspect = Number.isFinite(dmin) ? dmax / dmin : 1;

  // Holes ≈ cylindrical surfaces (open cyl ≈ thru-hole, capped ≈ blind)
  if (desc.features.holes > 0) {
    out.push({
      kind: "through_holes",
      count: desc.features.holes,
      severity: desc.features.holes > 8 ? "medium" : "low",
      detail: `${desc.features.holes} cylindrical surfaces — likely drilled or bored`,
    });
  }
  if (desc.features.fillets > 0) {
    out.push({
      kind: "fillets",
      count: desc.features.fillets,
      severity: "low",
      detail: `${desc.features.fillets} toroidal surfaces — radius blends between adjacent faces`,
    });
  }
  if (desc.features.chamfers > 0) {
    out.push({
      kind: "chamfers",
      count: desc.features.chamfers,
      severity: "low",
      detail: `${desc.features.chamfers} conical surfaces — likely chamfers or tapered features`,
    });
  }
  // Sharp corners ≈ planar-heavy + low fillet count
  const sharpRatio = curv.planar - Math.min(curv.toroidal, 0.3);
  if (sharpRatio > 0.7 && desc.counts.faces > 4) {
    out.push({
      kind: "sharp_corners",
      count: Math.round(sharpRatio * desc.counts.edges),
      severity: sharpRatio > 0.85 ? "high" : "medium",
      detail: `${(sharpRatio * 100).toFixed(0)}% planar surfaces with few fillets — stress raisers likely`,
    });
  }
  // Thin walls — high aspect + low solid count
  if (aspect > 8 && desc.counts.solids <= 2) {
    out.push({
      kind: "thin_walls",
      count: 1,
      severity: aspect > 16 ? "high" : "medium",
      detail: `bbox aspect ${aspect.toFixed(1)}× — slender geometry, deflection risk`,
    });
  }
  // Deep pockets: many edges per face
  if (topo.edgeFaceRatio > 6) {
    out.push({
      kind: "deep_pockets",
      count: Math.round(topo.edgeFaceRatio - 5),
      severity: topo.edgeFaceRatio > 10 ? "high" : "medium",
      detail: `${topo.edgeFaceRatio.toFixed(1)} edges/face — complex pocket boundaries`,
    });
  }
  // Spline-heavy → freeform / undercuts likely
  if (curv.spline > 0.25) {
    out.push({
      kind: "freeform_undercuts",
      count: Math.round(curv.spline * desc.counts.faces),
      severity: curv.spline > 0.5 ? "high" : "medium",
      detail: `${(curv.spline * 100).toFixed(0)}% B-spline surfaces — likely 5-axis or molded`,
    });
  }
  if (out.length === 0) {
    out.push({
      kind: "nominal",
      count: 0,
      severity: "low",
      detail: "no manufacturability flags above threshold",
    });
  }
  return out;
}

function risk(desc: GeomDescriptor, topo: TopologyDescriptor, curv: CurvatureSignature, finds: FeatureFinding[]): RiskHeuristics {
  const sharp = finds.find((f) => f.kind === "sharp_corners");
  const thin = finds.find((f) => f.kind === "thin_walls");
  const free = finds.find((f) => f.kind === "freeform_undercuts");
  const pock = finds.find((f) => f.kind === "deep_pockets");

  const sevW = (s?: Severity) => s === "high" ? 1 : s === "medium" ? 0.55 : s ? 0.2 : 0;

  // Stress concentration: sharp corners + thin walls + few fillets
  const filletRelief = Math.min(curv.toroidal * 1.5, 0.5);
  const stress = Math.max(0, Math.min(1,
    0.55 * sevW(sharp?.severity) + 0.35 * sevW(thin?.severity) + 0.15 * (1 - filletRelief)
  ));
  // Thermal: thin walls + planar dominance + deep pockets
  const thermal = Math.max(0, Math.min(1,
    0.5 * sevW(thin?.severity) + 0.25 * curv.planar + 0.25 * sevW(pock?.severity)
  ));
  // Fabrication: freeform + deep pockets + many components
  const compNorm = Math.min(1, topo.components / 8);
  const fab = Math.max(0, Math.min(1,
    0.45 * sevW(free?.severity) + 0.35 * sevW(pock?.severity) + 0.20 * compNorm
  ));

  const notes: string[] = [];
  if (stress > 0.6) notes.push("High stress raisers — add fillets at sharp corners");
  if (thermal > 0.6) notes.push("Thin walls limit heat dissipation — consider ribs or chamfered cooling paths");
  if (fab > 0.6) notes.push("Freeform / pocket complexity — plan 5-axis tooling or sectioning");
  if (notes.length === 0) notes.push("Within nominal physical envelope");
  return { stressConcentration: stress, thermalRisk: thermal, fabricationDifficulty: fab, notes };
}

// 12-dim fab feature vector
function fabVector(desc: GeomDescriptor, topo: TopologyDescriptor, curv: CurvatureSignature, r: RiskHeuristics): number[] {
  const dims = bboxDims(desc);
  const dmax = Math.max(...dims, 1e-9);
  const v = [
    Math.min((desc.features.holes ?? 0) / 10, 1),
    curv.planar, curv.cylindrical, curv.conical,
    curv.spherical, curv.toroidal, curv.spline,
    Math.min(topo.edgeFaceRatio / 12, 1),
    Math.min(topo.cyclomatic / Math.max(desc.counts.edges, 1), 1),
    r.stressConcentration, r.thermalRisk, r.fabricationDifficulty,
  ];
  // normalize each to [0..1] (already roughly bounded)
  return v.map((x) => Math.max(0, Math.min(1, x)));
}

// 32-dim embedding via fixed hashed projection of (type, count) pairs.
function embed(report: ParseReport, fabVec: number[]): number[] {
  const D = 32;
  const e = new Array(D).fill(0);
  // Hash bytes
  const h = (s: string) => {
    let x = 2166136261 >>> 0;
    for (let i = 0; i < s.length; i++) {
      x ^= s.charCodeAt(i);
      x = Math.imul(x, 16777619);
    }
    return x >>> 0;
  };
  for (const [t, ids] of report.byType) {
    const hash = h(t);
    const idx = hash % D;
    const sign = (hash >> 8) & 1 ? 1 : -1;
    e[idx] += sign * Math.log1p(ids.length);
  }
  // Mix in fab vector (deterministic placement)
  for (let i = 0; i < fabVec.length; i++) {
    e[(i * 7) % D] += fabVec[i] * 1.5;
  }
  // L2 normalize
  let s2 = 0;
  for (const v of e) s2 += v * v;
  const n = Math.sqrt(s2) || 1;
  return e.map((v) => v / n);
}

export function analyzeGeometry(
  report: ParseReport, topo: TopoGraph, desc: GeomDescriptor,
): FeatureIntelligence {
  const t0 = performance.now();
  const top = topologyDescriptor(report, topo, desc);
  const curv = curvatureSignature(desc, report);
  const finds = findings(desc, top, curv);
  const r = risk(desc, top, curv, finds);
  const fab = fabVector(desc, top, curv, r);
  const emb = embed(report, fab);
  return {
    topology: top, curvature: curv, findings: finds, risk: r,
    fabFeatureVector: fab, embedding: emb,
    ms: performance.now() - t0,
  };
}
