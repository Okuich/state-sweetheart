/**
 * Adaptive octree mesher.
 *
 * Builds a sparse octree over an axis-aligned bounding box, refining cells
 * whose density score (curvature, holes, fillets, hotspots, overhangs)
 * exceeds a refinement threshold. Each leaf is split into 6 tetrahedra so
 * downstream FEM/thermal solvers receive a tet mesh with surface-preserving
 * refinement near fabrication-critical features.
 *
 * Pure TypeScript, deterministic, no external deps. Designed to scale to
 * ~1e5 leaves on a Worker without exceeding heap.
 */

export type Vec3 = readonly [number, number, number];
export type AABB = { min: Vec3; max: Vec3 };

/** Refinement seeds: regions that should bias the octree to subdivide. */
export interface RefinementSeed {
  kind: "hole" | "fillet" | "sharp" | "hotspot" | "overhang" | "thin_wall" | "contact";
  /** Center in world coordinates. */
  center: Vec3;
  /** Influence radius (Gaussian falloff). */
  radius: number;
  /** Strength multiplier; higher = deeper refinement. */
  weight: number;
}

export interface OctreeOptions {
  /** Hard cap on subdivisions from root. */
  maxDepth: number;
  /** Minimum subdivisions everywhere (uniform base resolution). */
  minDepth: number;
  /** Cells with density >= this are split. */
  refineThreshold: number;
  /** Hard cap on total leaves to keep memory bounded. */
  maxLeaves: number;
}

export const DEFAULT_OCTREE_OPTIONS: OctreeOptions = {
  maxDepth: 6,
  minDepth: 2,
  refineThreshold: 0.35,
  maxLeaves: 200_000,
};

export interface OctreeNode {
  /** Global linear id (post-build). */
  id: number;
  depth: number;
  bbox: AABB;
  /** Per-leaf density score in [0,1]. */
  density: number;
  /** Children indices into nodes[], length 0 (leaf) or 8 (internal). */
  children: number[];
  /** True for leaves. */
  leaf: boolean;
  /** Dominant seed kind that drove refinement (for color/label). */
  tag: RefinementSeed["kind"] | "bulk";
}

export interface OctreeMesh {
  bbox: AABB;
  options: OctreeOptions;
  nodes: OctreeNode[];
  leaves: number[];
  /** Vertex pool, deduplicated to vertex resolution. */
  vertices: Float32Array;
  /** Tetrahedral connectivity (4 vertex indices per tet). */
  tets: Uint32Array;
  /** Parent leaf index per tet (for adjacency / partitioning). */
  tetLeaf: Uint32Array;
  /** Per-leaf surface-preservation flag (boundary touching seed). */
  boundaryLeaf: Uint8Array;
  buildMs: number;
}

const SUB_OFFSETS: Vec3[] = [
  [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
  [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
];

function aabbCenter(b: AABB): Vec3 {
  return [(b.min[0] + b.max[0]) / 2, (b.min[1] + b.max[1]) / 2, (b.min[2] + b.max[2]) / 2];
}

function aabbExtent(b: AABB): number {
  return Math.max(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]);
}

function subBox(b: AABB, off: Vec3): AABB {
  const c = aabbCenter(b);
  return {
    min: [
      off[0] ? c[0] : b.min[0],
      off[1] ? c[1] : b.min[1],
      off[2] ? c[2] : b.min[2],
    ],
    max: [
      off[0] ? b.max[0] : c[0],
      off[1] ? b.max[1] : c[1],
      off[2] ? b.max[2] : c[2],
    ],
  };
}

/** Density at a point given seeds, in [0,1] (sum of clamped Gaussians). */
export function seedDensity(p: Vec3, seeds: RefinementSeed[]): { d: number; tag: RefinementSeed["kind"] | "bulk" } {
  let sum = 0;
  let bestW = 0;
  let bestTag: RefinementSeed["kind"] | "bulk" = "bulk";
  for (const s of seeds) {
    const dx = p[0] - s.center[0];
    const dy = p[1] - s.center[1];
    const dz = p[2] - s.center[2];
    const r2 = dx * dx + dy * dy + dz * dz;
    const sigma2 = Math.max(1e-9, s.radius * s.radius);
    const g = Math.exp(-r2 / sigma2) * s.weight;
    sum += g;
    if (g > bestW) { bestW = g; bestTag = s.kind; }
  }
  return { d: Math.min(1, sum), tag: bestTag };
}

interface VertexPool {
  push: (p: Vec3) => number;
  toArray: () => Float32Array;
  count: () => number;
}

function makeVertexPool(quantum: number): VertexPool {
  const map = new Map<string, number>();
  const xs: number[] = [];
  const q = Math.max(1e-9, quantum);
  return {
    push: (p) => {
      const kx = Math.round(p[0] / q);
      const ky = Math.round(p[1] / q);
      const kz = Math.round(p[2] / q);
      const key = `${kx},${ky},${kz}`;
      const hit = map.get(key);
      if (hit !== undefined) return hit;
      const id = xs.length / 3;
      xs.push(kx * q, ky * q, kz * q);
      map.set(key, id);
      return id;
    },
    toArray: () => new Float32Array(xs),
    count: () => xs.length / 3,
  };
}

/** Decompose a hex (8 corners) into 6 tetrahedra (canonical Kuhn split). */
function hexToTets(c: number[]): number[] {
  // c indexed as SUB_OFFSETS order: 0..7
  return [
    c[0], c[1], c[3], c[7],
    c[0], c[3], c[2], c[7],
    c[0], c[2], c[6], c[7],
    c[0], c[6], c[4], c[7],
    c[0], c[4], c[5], c[7],
    c[0], c[5], c[1], c[7],
  ];
}

function corners(b: AABB): Vec3[] {
  return SUB_OFFSETS.map<Vec3>((o) => [
    o[0] ? b.max[0] : b.min[0],
    o[1] ? b.max[1] : b.min[1],
    o[2] ? b.max[2] : b.min[2],
  ]);
}

export function buildOctreeMesh(
  bbox: AABB,
  seeds: RefinementSeed[],
  optsIn: Partial<OctreeOptions> = {},
): OctreeMesh {
  const t0 = Date.now();
  const options: OctreeOptions = { ...DEFAULT_OCTREE_OPTIONS, ...optsIn };
  const nodes: OctreeNode[] = [];

  const root: OctreeNode = {
    id: 0,
    depth: 0,
    bbox,
    density: 0,
    children: [],
    leaf: true,
    tag: "bulk",
  };
  nodes.push(root);

  // Iterative refinement queue (BFS) — deterministic.
  const queue: number[] = [0];
  let leafCount = 1;

  while (queue.length) {
    const idx = queue.shift()!;
    const n = nodes[idx];
    const center = aabbCenter(n.bbox);
    const { d, tag } = seedDensity(center, seeds);
    n.density = d;
    n.tag = d > 0 ? tag : "bulk";
    const shouldRefine =
      n.depth < options.minDepth ||
      (n.depth < options.maxDepth && d >= options.refineThreshold);
    if (!shouldRefine) continue;
    if (leafCount + 7 > options.maxLeaves) continue;

    n.leaf = false;
    leafCount += 7;
    for (const off of SUB_OFFSETS) {
      const child: OctreeNode = {
        id: nodes.length,
        depth: n.depth + 1,
        bbox: subBox(n.bbox, off),
        density: 0,
        children: [],
        leaf: true,
        tag: "bulk",
      };
      n.children.push(child.id);
      nodes.push(child);
      queue.push(child.id);
    }
  }

  // Collect leaves + tetrahedralize.
  const leaves: number[] = [];
  for (const n of nodes) if (n.leaf) leaves.push(n.id);

  // Vertex pool quantized to a safe fraction of the smallest leaf edge.
  const ext = aabbExtent(bbox);
  const minLeafEdge = ext / Math.pow(2, options.maxDepth);
  const pool = makeVertexPool(minLeafEdge / 8);

  const tetIdx: number[] = [];
  const tetLeaf: number[] = [];
  const boundaryLeaf = new Uint8Array(leaves.length);

  leaves.forEach((leafId, lIdx) => {
    const n = nodes[leafId];
    const cs = corners(n.bbox).map((p) => pool.push(p));
    const tets = hexToTets(cs);
    for (let i = 0; i < 6; i++) {
      tetIdx.push(tets[i * 4 + 0], tets[i * 4 + 1], tets[i * 4 + 2], tets[i * 4 + 3]);
      tetLeaf.push(lIdx);
    }
    if (n.density >= options.refineThreshold * 0.6 || n.tag !== "bulk") {
      boundaryLeaf[lIdx] = 1;
    }
  });

  return {
    bbox,
    options,
    nodes,
    leaves,
    vertices: pool.toArray(),
    tets: new Uint32Array(tetIdx),
    tetLeaf: new Uint32Array(tetLeaf),
    boundaryLeaf,
    buildMs: Date.now() - t0,
  };
}
