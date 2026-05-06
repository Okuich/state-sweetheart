// Retrieval-Augmented Physics Reasoning
// ─────────────────────────────────────────────────────────────
// Builds a typed retrieval layer over the persistent knowledge
// graph + a local "episodic memory" of historical failures and
// optimization steps. Four retrievers:
//
//   • geometryKNN   — nearest geometry motifs by cosine
//   • topologyMatch — Jaccard over topological signatures
//   • failureLookup — past failure outcomes most similar to query
//   • optimMemory   — past optimization tweaks ranked by gain
//
// All in-memory / localStorage; no network.

import {
  cosine,
  loadGraph,
  type Graph,
  type GraphNode,
} from "./knowledgeGraph";

// ─── Query ───────────────────────────────────────────────────
export type RagQuery = {
  // free-form embedding (8-dim, same space as graph nodes)
  embedding: number[];
  // structural signature: a multiset of motif tokens
  topology: string[];
  // textual hint (used to bias re-ranking)
  hint?: string;
  // problem context
  context?: {
    targetStress?: number;   // MPa
    materialClass?: string;  // "metal" | "polymer" | "composite"
    process?: string;        // "mill-3ax" | "FDM" | …
  };
};

export type Retrieved<T> = {
  item: T;
  score: number;
  reason: string;
};

// ─── helpers ─────────────────────────────────────────────────
function jaccard(a: string[], b: string[]): number {
  if (a.length === 0 && b.length === 0) return 1;
  const A = new Set(a);
  const B = new Set(b);
  let inter = 0;
  for (const x of A) if (B.has(x)) inter += 1;
  const union = A.size + B.size - inter;
  return union === 0 ? 0 : inter / union;
}

function topoSignature(n: GraphNode): string[] {
  // Build a stable multiset of "tokens" from attrs + label.
  const tokens: string[] = [n.kind];
  for (const [k, v] of Object.entries(n.attrs)) {
    if (typeof v === "number") {
      // bucketize numbers
      const bucket =
        v < 0.5 ? "xs" :
        v < 2   ? "s"  :
        v < 10  ? "m"  :
        v < 100 ? "l"  : "xl";
      tokens.push(`${k}:${bucket}`);
    } else {
      tokens.push(`${k}:${String(v).toLowerCase()}`);
    }
  }
  for (const part of n.label.toLowerCase().split(/[-_\s/]+/)) {
    if (part) tokens.push(`tok:${part}`);
  }
  return tokens;
}

// ─── public retrievers ───────────────────────────────────────

export function geometryKNN(q: RagQuery, g: Graph, k = 5): Retrieved<GraphNode>[] {
  return g.nodes
    .filter((n) => n.kind === "geometry")
    .map((n) => ({
      item: n,
      score: cosine(q.embedding, n.embedding),
      reason: "cosine over 8-dim geometry embedding",
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
}

export function topologyMatch(q: RagQuery, g: Graph, k = 5): Retrieved<GraphNode>[] {
  return g.nodes
    .filter((n) => n.kind === "geometry")
    .map((n) => ({
      item: n,
      score: jaccard(q.topology, topoSignature(n)),
      reason: "Jaccard over topology tokens",
    }))
    .filter((r) => r.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
}

export function failureLookup(q: RagQuery, g: Graph, k = 5): Retrieved<GraphNode>[] {
  // failures = outcome nodes with pass=0
  const failures = g.nodes.filter(
    (n) => n.kind === "outcome" && Number(n.attrs.pass ?? 1) === 0
  );
  // re-rank by similarity of upstream sim's stress to target, then by
  // embedding cosine. Use causal edges to find linked sim node.
  return failures
    .map((f) => {
      const causalIn = g.edges.filter(
        (e) => e.kind === "causal" && e.to === f.id
      );
      const linkedSim = causalIn
        .map((e) => g.nodes.find((n) => n.id === e.from))
        .find((n) => n?.kind === "sim");

      let stressTerm = 0;
      if (linkedSim && q.context?.targetStress !== undefined) {
        const s = Number(linkedSim.attrs.maxStress ?? 0);
        stressTerm = 1 / (1 + Math.abs(s - q.context.targetStress) / 100);
      }
      const cosTerm = cosine(q.embedding, f.embedding);
      const defectTerm = Math.min(1, Number(f.attrs.defectRate ?? 0));
      const score = 0.5 * cosTerm + 0.3 * stressTerm + 0.2 * defectTerm;
      return {
        item: f,
        score,
        reason: linkedSim
          ? `sim=${linkedSim.label} σ=${linkedSim.attrs.maxStress}MPa, defect=${f.attrs.defectRate}`
          : "no upstream sim",
      };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
}

export type OptimRecord = {
  parent: GraphNode;
  child: GraphNode;
  note: string;
  weight: number;
};

export function optimMemory(q: RagQuery, g: Graph, k = 5): Retrieved<OptimRecord>[] {
  const recs: Retrieved<OptimRecord>[] = [];
  for (const e of g.edges) {
    if (e.kind !== "optimized-from") continue;
    const parent = g.nodes.find((n) => n.id === e.from);
    const child  = g.nodes.find((n) => n.id === e.to);
    if (!parent || !child) continue;
    const sim = cosine(q.embedding, parent.embedding);
    recs.push({
      item: { parent, child, note: e.note ?? "", weight: e.weight },
      score: 0.7 * sim + 0.3 * e.weight,
      reason: `parent-sim=${sim.toFixed(2)} edge-w=${e.weight.toFixed(2)}`,
    });
  }
  return recs.sort((a, b) => b.score - a.score).slice(0, k);
}

// ─── compose context for downstream reasoner ────────────────
export type RagContext = {
  query: RagQuery;
  geometry: Retrieved<GraphNode>[];
  topology: Retrieved<GraphNode>[];
  failures: Retrieved<GraphNode>[];
  optimizations: Retrieved<OptimRecord>[];
  summary: string;
};

export function retrieveAll(q: RagQuery, g: Graph = loadGraph(), k = 5): RagContext {
  const geometry      = geometryKNN(q, g, k);
  const topology      = topologyMatch(q, g, k);
  const failures      = failureLookup(q, g, k);
  const optimizations = optimMemory(q, g, k);

  const summary = [
    `K=${k} retrievals over graph(${g.nodes.length} nodes, ${g.edges.length} edges)`,
    geometry[0]   && `top geom: ${geometry[0].item.label}   (${geometry[0].score.toFixed(2)})`,
    topology[0]   && `top topo: ${topology[0].item.label}   (${topology[0].score.toFixed(2)})`,
    failures[0]   && `top fail: ${failures[0].item.label}   (${failures[0].score.toFixed(2)})`,
    optimizations[0] && `top tweak: ${optimizations[0].item.note}`,
  ]
    .filter(Boolean)
    .join(" · ");

  return { query: q, geometry, topology, failures, optimizations, summary };
}

// ─── synthesize a query from a free-form description ────────
function hash32(s: string) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}
export function synthesizeQuery(text: string, ctx?: RagQuery["context"]): RagQuery {
  const tokens = text.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
  // 8-dim deterministic embedding from token bag
  const v = new Array(8).fill(0);
  for (const t of tokens) {
    const h = hash32(t);
    for (let i = 0; i < 8; i++) {
      v[i] += ((h >> (i * 3)) & 0xff) / 255 - 0.5;
    }
  }
  const n = Math.sqrt(v.reduce((a, x) => a + x * x, 0)) || 1;
  return {
    embedding: v.map((x) => x / n),
    topology: tokens.map((t) => `tok:${t}`),
    hint: text,
    context: ctx,
  };
}
