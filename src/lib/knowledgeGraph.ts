// Geometry-Physics Knowledge Graph
// ─────────────────────────────────────────────────────────────
// Persistent (localStorage-backed) graph linking:
//   • geometry motifs   (thin-wall, deep-pocket, bracket, lattice, …)
//   • material systems  (Al-7075, SS-316, CFRP, ABS, Ti-6Al-4V, …)
//   • process params    (mill-3ax, mill-5ax, SLA, FDM, SLS, …)
//   • simulation states (sim runs with energy/stress summaries)
//   • fab outcomes      (pass/fail, defect rate, dim. accuracy)
// Edges:
//   • causal     (param → outcome, geometry+material → stress)
//   • similarity (cosine over node embeddings)
//   • optimized-from (history: parent → child along a tweak)

export type NodeKind =
  | "geometry"
  | "material"
  | "process"
  | "sim"
  | "outcome";

export type EdgeKind = "causal" | "similar" | "optimized-from";

export type GraphNode = {
  id: string;
  kind: NodeKind;
  label: string;
  embedding: number[]; // 8-dim
  attrs: Record<string, number | string>;
  createdAt: number;
};

export type GraphEdge = {
  id: string;
  kind: EdgeKind;
  from: string;
  to: string;
  weight: number; // 0..1
  note?: string;
  createdAt: number;
};

export type Graph = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  version: number;
};

const STORAGE_KEY = "physics.knowledgeGraph.v1";

export function emptyGraph(): Graph {
  return { nodes: [], edges: [], version: 1 };
}

export function loadGraph(): Graph {
  if (typeof window === "undefined") return emptyGraph();
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return emptyGraph();
    const g = JSON.parse(raw) as Graph;
    if (!g || !Array.isArray(g.nodes) || !Array.isArray(g.edges)) return emptyGraph();
    return g;
  } catch {
    return emptyGraph();
  }
}

export function saveGraph(g: Graph) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(g));
  } catch {
    /* ignore quota */
  }
}

// ─── ids ─────────────────────────────────────────────────────
let _seq = 0;
function uid(prefix: string) {
  _seq += 1;
  return `${prefix}_${Date.now().toString(36)}_${_seq.toString(36)}`;
}

// ─── embedding helpers ───────────────────────────────────────
function hashStr(s: string): number {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function seededVec(seed: string, dim = 8): number[] {
  let h = hashStr(seed);
  const v: number[] = [];
  for (let i = 0; i < dim; i++) {
    h = Math.imul(h ^ (h >>> 13), 0x5bd1e995);
    v.push(((h >>> 0) / 0xffffffff) * 2 - 1);
  }
  // normalize
  const n = Math.sqrt(v.reduce((a, x) => a + x * x, 0)) || 1;
  return v.map((x) => x / n);
}

export function cosine(a: number[], b: number[]): number {
  const m = Math.min(a.length, b.length);
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < m; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

// ─── node / edge mutation ────────────────────────────────────
export function addNode(
  g: Graph,
  kind: NodeKind,
  label: string,
  attrs: Record<string, number | string> = {}
): GraphNode {
  const node: GraphNode = {
    id: uid(kind),
    kind,
    label,
    embedding: seededVec(`${kind}:${label}:${JSON.stringify(attrs)}`),
    attrs,
    createdAt: Date.now(),
  };
  g.nodes.push(node);
  return node;
}

export function addEdge(
  g: Graph,
  kind: EdgeKind,
  from: string,
  to: string,
  weight: number,
  note?: string
): GraphEdge {
  const edge: GraphEdge = {
    id: uid("e"),
    kind,
    from,
    to,
    weight: Math.max(0, Math.min(1, weight)),
    note,
    createdAt: Date.now(),
  };
  g.edges.push(edge);
  return edge;
}

export function removeNode(g: Graph, id: string) {
  g.nodes = g.nodes.filter((n) => n.id !== id);
  g.edges = g.edges.filter((e) => e.from !== id && e.to !== id);
}

// ─── similarity recompute ────────────────────────────────────
// Recomputes "similar" edges within a node-kind cluster above threshold.
export function recomputeSimilarity(g: Graph, threshold = 0.55) {
  g.edges = g.edges.filter((e) => e.kind !== "similar");
  const byKind = new Map<NodeKind, GraphNode[]>();
  for (const n of g.nodes) {
    const arr = byKind.get(n.kind) ?? [];
    arr.push(n);
    byKind.set(n.kind, arr);
  }
  for (const [, list] of byKind) {
    for (let i = 0; i < list.length; i++) {
      for (let j = i + 1; j < list.length; j++) {
        const w = cosine(list[i].embedding, list[j].embedding);
        if (w >= threshold) {
          addEdge(g, "similar", list[i].id, list[j].id, w, "cosine");
        }
      }
    }
  }
}

// ─── seed industrial corpus ──────────────────────────────────
export function seedIndustrial(g: Graph) {
  const motifs = [
    ["thin-wall-bracket",   { thickness: 1.2, aspect: 8 }],
    ["deep-pocket-housing", { depth: 40,      aspect: 5 }],
    ["lattice-cube",        { cellSize: 4,    porosity: 0.62 }],
    ["fillet-arm",          { minR: 0.8,      length: 80 }],
    ["thin-rib-array",      { thickness: 0.9, ribCount: 12 }],
  ] as const;

  const mats = [
    ["Al-7075", { E: 71.7, sigmaY: 503, rho: 2.81 }],
    ["SS-316",  { E: 193,  sigmaY: 290, rho: 7.99 }],
    ["Ti-6Al-4V", { E: 113.8, sigmaY: 880, rho: 4.43 }],
    ["CFRP",    { E: 70,   sigmaY: 600, rho: 1.55 }],
    ["ABS",     { E: 2.3,  sigmaY: 40,  rho: 1.05 }],
  ] as const;

  const procs = [
    ["mill-3ax", { feed: 800,  spindle: 12000, cool: "flood" }],
    ["mill-5ax", { feed: 1200, spindle: 18000, cool: "mist"  }],
    ["SLA",      { layer: 0.05, exposure: 1.4 }],
    ["FDM",      { layer: 0.2,  nozzle: 0.4   }],
    ["SLS",      { layer: 0.1,  laser: 30     }],
  ] as const;

  const motifNodes = motifs.map(([l, a]) => addNode(g, "geometry", l, a as Record<string, number | string>));
  const matNodes   = mats.map(([l, a])   => addNode(g, "material", l, a as Record<string, number | string>));
  const procNodes  = procs.map(([l, a])  => addNode(g, "process",  l, a as Record<string, number | string>));

  // sim runs + outcomes (causal chain)
  const rng = (() => { let s = 1337; return () => (s = (s * 16807) % 2147483647) / 2147483647; })();
  for (let i = 0; i < 8; i++) {
    const m = motifNodes[Math.floor(rng() * motifNodes.length)];
    const mat = matNodes[Math.floor(rng() * matNodes.length)];
    const proc = procNodes[Math.floor(rng() * procNodes.length)];

    const stress = Math.round(50 + rng() * 350);
    const energyDrift = +(rng() * 0.02).toFixed(4);
    const sim = addNode(g, "sim", `run-${i + 1}`, {
      maxStress: stress,
      energyDrift,
      steps: 5000 + Math.floor(rng() * 5000),
    });
    addEdge(g, "causal", m.id,   sim.id, 0.7, "geometry → stress");
    addEdge(g, "causal", mat.id, sim.id, 0.6, "material → stiffness");

    const ok = stress < 320 && rng() > 0.2;
    const outcome = addNode(g, "outcome", `${proc.label}/${ok ? "pass" : "fail"}-${i + 1}`, {
      pass: ok ? 1 : 0,
      defectRate: +(ok ? rng() * 0.04 : 0.05 + rng() * 0.2).toFixed(3),
      dimError_um: Math.round(10 + rng() * (ok ? 30 : 120)),
    });
    addEdge(g, "causal", proc.id, outcome.id, 0.8, "process → outcome");
    addEdge(g, "causal", sim.id,  outcome.id, 0.55, "predicted → measured");
  }

  // optimization history: take a motif and create an "optimized" child
  const parent = motifNodes[0];
  const child = addNode(g, "geometry", `${parent.label}*`, {
    ...parent.attrs,
    thickness: Number(parent.attrs.thickness ?? 1) + 0.4,
    note: "thickened for stress",
  });
  addEdge(g, "optimized-from", parent.id, child.id, 0.9, "+thickness 0.4mm");

  recomputeSimilarity(g);
}

// ─── analytics ───────────────────────────────────────────────
export type GraphStats = {
  nodes: number;
  edges: number;
  byKind: Record<NodeKind, number>;
  byEdge: Record<EdgeKind, number>;
  avgDegree: number;
  density: number;
  components: number;
};

export function stats(g: Graph): GraphStats {
  const byKind = { geometry: 0, material: 0, process: 0, sim: 0, outcome: 0 } as Record<NodeKind, number>;
  const byEdge = { causal: 0, similar: 0, "optimized-from": 0 } as Record<EdgeKind, number>;
  for (const n of g.nodes) byKind[n.kind] += 1;
  for (const e of g.edges) byEdge[e.kind] += 1;

  const adj = new Map<string, Set<string>>();
  for (const n of g.nodes) adj.set(n.id, new Set());
  for (const e of g.edges) {
    adj.get(e.from)?.add(e.to);
    adj.get(e.to)?.add(e.from);
  }
  const degSum = [...adj.values()].reduce((a, s) => a + s.size, 0);
  const N = g.nodes.length || 1;
  const possible = (N * (N - 1)) / 2 || 1;

  // connected components (undirected)
  const seen = new Set<string>();
  let comps = 0;
  for (const n of g.nodes) {
    if (seen.has(n.id)) continue;
    comps += 1;
    const stack = [n.id];
    while (stack.length) {
      const v = stack.pop()!;
      if (seen.has(v)) continue;
      seen.add(v);
      for (const u of adj.get(v) ?? []) if (!seen.has(u)) stack.push(u);
    }
  }

  return {
    nodes: g.nodes.length,
    edges: g.edges.length,
    byKind,
    byEdge,
    avgDegree: degSum / N,
    density: g.edges.length / possible,
    components: comps,
  };
}

// k nearest neighbors (any kind) by cosine
export function knn(g: Graph, nodeId: string, k = 5) {
  const me = g.nodes.find((n) => n.id === nodeId);
  if (!me) return [];
  return g.nodes
    .filter((n) => n.id !== nodeId)
    .map((n) => ({ node: n, sim: cosine(me.embedding, n.embedding) }))
    .sort((a, b) => b.sim - a.sim)
    .slice(0, k);
}

// causal path: BFS over "causal" edges only
export function causalPath(g: Graph, fromId: string, toId: string): string[] {
  const adj = new Map<string, string[]>();
  for (const e of g.edges) {
    if (e.kind !== "causal") continue;
    const a = adj.get(e.from) ?? [];
    a.push(e.to);
    adj.set(e.from, a);
  }
  const prev = new Map<string, string | null>();
  prev.set(fromId, null);
  const q = [fromId];
  while (q.length) {
    const v = q.shift()!;
    if (v === toId) break;
    for (const u of adj.get(v) ?? []) {
      if (!prev.has(u)) {
        prev.set(u, v);
        q.push(u);
      }
    }
  }
  if (!prev.has(toId)) return [];
  const path: string[] = [];
  let cur: string | null | undefined = toId;
  while (cur) {
    path.unshift(cur);
    cur = prev.get(cur) ?? null;
  }
  return path;
}
