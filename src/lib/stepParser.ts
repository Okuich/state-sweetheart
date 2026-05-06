// STEP File Ingestion Engine
// ─────────────────────────────────────────────────────────────
// Lightweight STEP (ISO-10303) Part 21 parser → canonical
// representation: entities, topology graph, manifold/feature
// summary. Validates + repairs (orphans, degenerate edges,
// tolerance bands) and emits a normalized JSON form.
//
// Scope: AP203 / AP214 / AP242 entity vocabulary (CARTESIAN_POINT,
// VERTEX_POINT, EDGE_CURVE, ORIENTED_EDGE, FACE_BOUND, ADVANCED_FACE,
// CLOSED_SHELL, MANIFOLD_SOLID_BREP, …). Geometry math is not
// re-implemented — we extract structure, not surface evaluation.

export type EntityRef = number; // #123

export interface StepEntity {
  id: EntityRef;
  type: string;
  args: StepArg[];
  raw: string;
}

export type StepArg =
  | { kind: "ref"; ref: EntityRef }
  | { kind: "num"; value: number }
  | { kind: "str"; value: string }
  | { kind: "enum"; value: string }      // .T. .F. .UNSPECIFIED.
  | { kind: "list"; items: StepArg[] }
  | { kind: "null" };

export interface StepHeader {
  fileName?: string;
  fileSchema?: string;     // AP203 / AP214 / AP242
  description?: string;
  author?: string;
  organization?: string;
  timestamp?: string;
}

export interface ParseReport {
  header: StepHeader;
  entities: Map<EntityRef, StepEntity>;
  byType: Map<string, EntityRef[]>;
  errors: string[];
  warnings: string[];
  durationMs: number;
}

const ENTITY_LINE = /^#(\d+)\s*=\s*([A-Z_][A-Z0-9_]*)\s*\((.*)\)\s*;?\s*$/;

// ── Tokenizer for argument list ──
function parseArgs(src: string): StepArg[] {
  const out: StepArg[] = [];
  let i = 0;
  const n = src.length;

  const skipWs = () => { while (i < n && /\s/.test(src[i])) i++; };

  const parseOne = (): StepArg => {
    skipWs();
    const ch = src[i];
    if (ch === undefined) return { kind: "null" };
    if (ch === "$") { i++; return { kind: "null" }; }
    if (ch === "*") { i++; return { kind: "null" }; }
    if (ch === "#") {
      i++;
      let s = "";
      while (i < n && /[0-9]/.test(src[i])) { s += src[i++]; }
      return { kind: "ref", ref: parseInt(s, 10) };
    }
    if (ch === "'") {
      i++;
      let s = "";
      while (i < n) {
        if (src[i] === "'" && src[i + 1] === "'") { s += "'"; i += 2; continue; }
        if (src[i] === "'") { i++; break; }
        s += src[i++];
      }
      return { kind: "str", value: s };
    }
    if (ch === ".") {
      i++;
      let s = "";
      while (i < n && src[i] !== ".") s += src[i++];
      if (src[i] === ".") i++;
      return { kind: "enum", value: s };
    }
    if (ch === "(") {
      i++;
      const items: StepArg[] = [];
      skipWs();
      if (src[i] !== ")") {
        items.push(parseOne());
        skipWs();
        while (src[i] === ",") { i++; items.push(parseOne()); skipWs(); }
      }
      if (src[i] === ")") i++;
      return { kind: "list", items };
    }
    if (/[A-Z_a-z]/.test(ch)) {
      // Typed constructor (e.g., LENGTH_MEASURE(1.0)) — skip name, keep payload
      let name = "";
      while (i < n && /[A-Z_a-z0-9]/.test(src[i])) { name += src[i++]; }
      skipWs();
      if (src[i] === "(") {
        const list = parseOne(); // parses (...)
        if (list.kind === "list" && list.items.length === 1) return list.items[0];
        return list;
      }
      return { kind: "enum", value: name };
    }
    // number
    let s = "";
    while (i < n && /[\d+\-eE.]/.test(src[i])) { s += src[i++]; }
    const v = parseFloat(s);
    if (Number.isFinite(v)) return { kind: "num", value: v };
    return { kind: "null" };
  };

  skipWs();
  if (i >= n) return out;
  out.push(parseOne());
  skipWs();
  while (src[i] === ",") { i++; out.push(parseOne()); skipWs(); }
  return out;
}

export function parseStep(text: string): ParseReport {
  const t0 = performance.now();
  const errors: string[] = [];
  const warnings: string[] = [];
  const header: StepHeader = {};
  const entities = new Map<EntityRef, StepEntity>();
  const byType = new Map<string, EntityRef[]>();

  // Strip block comments /* ... */
  const stripped = text.replace(/\/\*[\s\S]*?\*\//g, " ");

  // Split into HEADER / DATA sections
  const headerMatch = stripped.match(/HEADER;([\s\S]*?)ENDSEC;/i);
  if (headerMatch) {
    const hsec = headerMatch[1];
    const fName = hsec.match(/FILE_NAME\s*\(([\s\S]*?)\)\s*;/i);
    const fSchema = hsec.match(/FILE_SCHEMA\s*\(\s*\(\s*'([^']+)'/i);
    const fDesc = hsec.match(/FILE_DESCRIPTION\s*\(\s*\(\s*'([^']+)'/i);
    if (fSchema) header.fileSchema = fSchema[1];
    if (fDesc) header.description = fDesc[1];
    if (fName) {
      const args = parseArgs(fName[1]);
      if (args[0]?.kind === "str") header.fileName = args[0].value;
      if (args[1]?.kind === "str") header.timestamp = args[1].value;
      const list2 = args[2];
      if (list2?.kind === "list" && list2.items[0]?.kind === "str") header.author = list2.items[0].value;
      const list3 = args[3];
      if (list3?.kind === "list" && list3.items[0]?.kind === "str") header.organization = list3.items[0].value;
    }
  } else {
    warnings.push("No HEADER; section found");
  }

  // DATA section: gather entity lines (multiline-safe)
  const dataMatch = stripped.match(/DATA;([\s\S]*?)ENDSEC;/i);
  const dataSrc = dataMatch ? dataMatch[1] : stripped;

  // Re-join lines so each statement ends with ';'
  const stmts: string[] = [];
  let buf = "";
  for (const ch of dataSrc) {
    buf += ch;
    if (ch === ";") {
      const s = buf.trim();
      if (s) stmts.push(s);
      buf = "";
    }
  }

  for (const stmt of stmts) {
    const m = stmt.match(ENTITY_LINE);
    if (!m) {
      // Could be a complex entity (a, b, c) syntax; flag as warning but skip
      if (/^#\d+\s*=/.test(stmt)) warnings.push(`Skipped complex entity: ${stmt.slice(0, 60)}…`);
      continue;
    }
    const id = parseInt(m[1], 10);
    const type = m[2].toUpperCase();
    const args = parseArgs(m[3]);
    if (entities.has(id)) {
      errors.push(`Duplicate entity #${id}`);
      continue;
    }
    const ent: StepEntity = { id, type, args, raw: stmt };
    entities.set(id, ent);
    const list = byType.get(type) ?? [];
    list.push(id);
    byType.set(type, list);
  }

  if (entities.size === 0) errors.push("No entities parsed");

  return {
    header, entities, byType, errors, warnings,
    durationMs: performance.now() - t0,
  };
}

// ── Topology graph ──
export interface TopoNode { id: EntityRef; type: string; refs: EntityRef[]; }
export interface TopoGraph {
  nodes: Map<EntityRef, TopoNode>;
  outEdges: Map<EntityRef, EntityRef[]>;
  inEdges: Map<EntityRef, EntityRef[]>;
  roots: EntityRef[];   // entities with no in-edges
  orphans: EntityRef[]; // referenced but missing
}

export function buildTopology(report: ParseReport): TopoGraph {
  const nodes = new Map<EntityRef, TopoNode>();
  const outEdges = new Map<EntityRef, EntityRef[]>();
  const inEdges = new Map<EntityRef, EntityRef[]>();
  const orphans = new Set<EntityRef>();

  const collectRefs = (a: StepArg, into: EntityRef[]) => {
    if (a.kind === "ref") into.push(a.ref);
    else if (a.kind === "list") a.items.forEach((it) => collectRefs(it, into));
  };

  for (const ent of report.entities.values()) {
    const refs: EntityRef[] = [];
    ent.args.forEach((a) => collectRefs(a, refs));
    nodes.set(ent.id, { id: ent.id, type: ent.type, refs });
    outEdges.set(ent.id, refs);
    for (const r of refs) {
      if (!report.entities.has(r)) orphans.add(r);
      const arr = inEdges.get(r) ?? [];
      arr.push(ent.id);
      inEdges.set(r, arr);
    }
  }
  const roots: EntityRef[] = [];
  for (const id of nodes.keys()) {
    if (!inEdges.has(id) || inEdges.get(id)!.length === 0) roots.push(id);
  }
  return { nodes, outEdges, inEdges, roots, orphans: [...orphans] };
}

// ── Canonical descriptor ──
export interface GeomDescriptor {
  schema: string;
  counts: {
    points: number; vertices: number; edges: number; faces: number;
    shells: number; solids: number; assemblies: number; surfaces: number;
  };
  manifold: boolean;
  bbox: { min: [number, number, number]; max: [number, number, number] } | null;
  features: { holes: number; fillets: number; chamfers: number; planar: number; cylindrical: number };
}

export function describe(report: ParseReport, topo: TopoGraph): GeomDescriptor {
  const C = (k: string) => report.byType.get(k)?.length ?? 0;
  const points = C("CARTESIAN_POINT");
  const counts = {
    points,
    vertices:   C("VERTEX_POINT"),
    edges:      C("EDGE_CURVE") + C("ORIENTED_EDGE"),
    faces:      C("ADVANCED_FACE") + C("FACE_SURFACE"),
    shells:     C("CLOSED_SHELL") + C("OPEN_SHELL"),
    solids:     C("MANIFOLD_SOLID_BREP") + C("BREP_WITH_VOIDS"),
    assemblies: C("NEXT_ASSEMBLY_USAGE_OCCURRENCE") + C("CONTEXT_DEPENDENT_SHAPE_REPRESENTATION"),
    surfaces:   C("PLANE") + C("CYLINDRICAL_SURFACE") + C("CONICAL_SURFACE")
              + C("SPHERICAL_SURFACE") + C("TOROIDAL_SURFACE") + C("B_SPLINE_SURFACE_WITH_KNOTS"),
  };

  // bbox over CARTESIAN_POINT
  let bbox: GeomDescriptor["bbox"] = null;
  const ptIds = report.byType.get("CARTESIAN_POINT") ?? [];
  if (ptIds.length > 0) {
    let mn: [number, number, number] = [Infinity, Infinity, Infinity];
    let mx: [number, number, number] = [-Infinity, -Infinity, -Infinity];
    for (const id of ptIds) {
      const ent = report.entities.get(id)!;
      const list = ent.args.find((a) => a.kind === "list");
      if (list?.kind !== "list") continue;
      const xyz = list.items.filter((x) => x.kind === "num").map((x) => (x as { value: number }).value);
      for (let k = 0; k < 3 && k < xyz.length; k++) {
        if (xyz[k] < mn[k]) mn[k] = xyz[k];
        if (xyz[k] > mx[k]) mx[k] = xyz[k];
      }
    }
    if (Number.isFinite(mn[0])) bbox = { min: mn, max: mx };
  }

  const manifold = counts.solids > 0 && topo.orphans.length === 0;

  const features = {
    holes:        C("CYLINDRICAL_SURFACE"),
    fillets:      C("TOROIDAL_SURFACE"),
    chamfers:     C("CONICAL_SURFACE"),
    planar:       C("PLANE"),
    cylindrical:  C("CYLINDRICAL_SURFACE"),
  };

  return {
    schema: report.header.fileSchema ?? "unknown",
    counts, manifold, bbox, features,
  };
}

// ── Validation ──
export interface Issue {
  severity: "error" | "warning" | "info";
  code: string;
  message: string;
  entity?: EntityRef;
}

export interface ValidationReport {
  ok: boolean;
  issues: Issue[];
  stats: { entities: number; orphans: number; duplicates: number; degenerate: number };
}

export function validate(report: ParseReport, topo: TopoGraph, desc: GeomDescriptor): ValidationReport {
  const issues: Issue[] = [];
  for (const ref of topo.orphans) {
    issues.push({ severity: "error", code: "orphan_ref", message: `Reference to missing #${ref}`, entity: ref });
  }
  for (const e of report.errors) issues.push({ severity: "error", code: "parse", message: e });
  for (const w of report.warnings) issues.push({ severity: "warning", code: "parse", message: w });

  if (desc.counts.solids === 0 && desc.counts.shells === 0) {
    issues.push({ severity: "warning", code: "no_solid", message: "No solid or shell entities — non-manifold representation" });
  }
  if (desc.counts.faces === 0) {
    issues.push({ severity: "warning", code: "no_face", message: "No face entities — geometry may be wireframe-only" });
  }

  // Degenerate edges: EDGE_CURVE whose two vertex refs are identical
  let degenerate = 0;
  for (const id of report.byType.get("EDGE_CURVE") ?? []) {
    const ent = report.entities.get(id)!;
    const refs = ent.args.filter((a) => a.kind === "ref") as Extract<StepArg, { kind: "ref" }>[];
    if (refs.length >= 2 && refs[0].ref === refs[1].ref) {
      degenerate++;
      issues.push({ severity: "warning", code: "degenerate_edge", message: `EDGE_CURVE #${id} has identical endpoints`, entity: id });
    }
  }

  return {
    ok: issues.every((x) => x.severity !== "error"),
    issues,
    stats: {
      entities: report.entities.size,
      orphans: topo.orphans.length,
      duplicates: 0,
      degenerate,
    },
  };
}

// ── Repair (in-memory) ──
export interface RepairResult {
  removedOrphans: number;
  collapsedDegenerate: number;
  toleranceBand: number;
  mergedPoints: number;
}

export function repair(report: ParseReport, topo: TopoGraph, tolerance = 1e-4): RepairResult {
  let removedOrphans = 0;
  // Strip arg refs that point to missing entities
  const missing = new Set(topo.orphans);
  for (const ent of report.entities.values()) {
    let changed = false;
    const stripRefs = (a: StepArg): StepArg => {
      if (a.kind === "ref" && missing.has(a.ref)) { changed = true; return { kind: "null" }; }
      if (a.kind === "list") return { kind: "list", items: a.items.map(stripRefs) };
      return a;
    };
    const newArgs = ent.args.map(stripRefs);
    if (changed) { ent.args = newArgs; removedOrphans++; }
  }

  // Drop degenerate EDGE_CURVEs
  let collapsed = 0;
  for (const id of report.byType.get("EDGE_CURVE") ?? []) {
    const ent = report.entities.get(id);
    if (!ent) continue;
    const refs = ent.args.filter((a) => a.kind === "ref") as Extract<StepArg, { kind: "ref" }>[];
    if (refs.length >= 2 && refs[0].ref === refs[1].ref) {
      report.entities.delete(id);
      collapsed++;
    }
  }
  if (collapsed > 0) {
    const list = (report.byType.get("EDGE_CURVE") ?? []).filter((id) => report.entities.has(id));
    report.byType.set("EDGE_CURVE", list);
  }

  // Tolerance-merge nearby CARTESIAN_POINT (record only — does not rewrite refs to keep this O(n))
  let merged = 0;
  const ptIds = report.byType.get("CARTESIAN_POINT") ?? [];
  const seen: { id: EntityRef; xyz: number[] }[] = [];
  for (const id of ptIds) {
    const ent = report.entities.get(id);
    if (!ent) continue;
    const list = ent.args.find((a) => a.kind === "list");
    if (list?.kind !== "list") continue;
    const xyz = list.items.filter((x) => x.kind === "num").map((x) => (x as { value: number }).value);
    let dup = false;
    for (const s of seen) {
      const d = Math.hypot(xyz[0] - s.xyz[0], xyz[1] - s.xyz[1], xyz[2] - s.xyz[2]);
      if (d < tolerance) { merged++; dup = true; break; }
    }
    if (!dup) seen.push({ id, xyz });
  }

  return {
    removedOrphans,
    collapsedDegenerate: collapsed,
    toleranceBand: tolerance,
    mergedPoints: merged,
  };
}

// ── Sample STEP fixture (small unit cube) ──
export const SAMPLE_STEP = `ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('Unit cube · canonical fixture'),'2;1');
FILE_NAME('cube.step','2026-05-06T12:00:00',('lovable'),('PhysicsOS'),'','','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#1 = APPLICATION_CONTEXT('mechanical design');
#10 = CARTESIAN_POINT('p0',(0.,0.,0.));
#11 = CARTESIAN_POINT('p1',(1.,0.,0.));
#12 = CARTESIAN_POINT('p2',(1.,1.,0.));
#13 = CARTESIAN_POINT('p3',(0.,1.,0.));
#14 = CARTESIAN_POINT('p4',(0.,0.,1.));
#15 = CARTESIAN_POINT('p5',(1.,0.,1.));
#16 = CARTESIAN_POINT('p6',(1.,1.,1.));
#17 = CARTESIAN_POINT('p7',(0.,1.,1.));
#20 = VERTEX_POINT('v0',#10);
#21 = VERTEX_POINT('v1',#11);
#22 = VERTEX_POINT('v2',#12);
#23 = VERTEX_POINT('v3',#13);
#24 = VERTEX_POINT('v4',#14);
#25 = VERTEX_POINT('v5',#15);
#26 = VERTEX_POINT('v6',#16);
#27 = VERTEX_POINT('v7',#17);
#30 = EDGE_CURVE('e0',#20,#21,$,.T.);
#31 = EDGE_CURVE('e1',#21,#22,$,.T.);
#32 = EDGE_CURVE('e2',#22,#23,$,.T.);
#33 = EDGE_CURVE('e3',#23,#20,$,.T.);
#34 = EDGE_CURVE('e4',#24,#25,$,.T.);
#35 = EDGE_CURVE('e5',#25,#26,$,.T.);
#36 = EDGE_CURVE('e6',#26,#27,$,.T.);
#37 = EDGE_CURVE('e7',#27,#24,$,.T.);
#38 = EDGE_CURVE('e8',#20,#24,$,.T.);
#39 = EDGE_CURVE('e9',#21,#25,$,.T.);
#40 = EDGE_CURVE('e10',#22,#26,$,.T.);
#41 = EDGE_CURVE('e11',#23,#27,$,.T.);
#50 = PLANE('bottom',#10);
#51 = PLANE('top',#14);
#52 = PLANE('side0',#10);
#53 = PLANE('side1',#11);
#54 = PLANE('side2',#12);
#55 = PLANE('side3',#13);
#60 = ADVANCED_FACE('f0',(#30,#31,#32,#33),#50,.T.);
#61 = ADVANCED_FACE('f1',(#34,#35,#36,#37),#51,.T.);
#62 = ADVANCED_FACE('f2',(#30,#39,#34,#38),#52,.T.);
#63 = ADVANCED_FACE('f3',(#31,#40,#35,#39),#53,.T.);
#64 = ADVANCED_FACE('f4',(#32,#41,#36,#40),#54,.T.);
#65 = ADVANCED_FACE('f5',(#33,#38,#37,#41),#55,.T.);
#70 = CLOSED_SHELL('shell',(#60,#61,#62,#63,#64,#65));
#80 = MANIFOLD_SOLID_BREP('cube',#70);
ENDSEC;
END-ISO-10303-21;
`;
