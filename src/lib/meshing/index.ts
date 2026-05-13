/**
 * Top-level meshing entry point.
 *
 * Pipes Geometry OS output (bbox + feature list) through:
 *   bbox + seeds  →  octree refine  →  tet split  →  quality eval
 *                    →  GPU adjacency  →  MPI partition + halo
 *
 * Returns a compact serializable summary safe to persist on step_jobs.mesh
 * plus the in-memory tensors for downstream Physics OS use.
 */

import {
  buildOctreeMesh,
  type AABB,
  type OctreeMesh,
  type OctreeOptions,
  type RefinementSeed,
} from "./octree";
import { evaluateQuality, type QualityReport } from "./quality";
import { buildAdjacency, type AdjacencyTensors } from "./adjacency";
import { partitionMesh, type PartitionPlan } from "./partition";

export type { AABB, RefinementSeed, OctreeOptions, OctreeMesh } from "./octree";
export type { QualityReport } from "./quality";
export type { AdjacencyTensors } from "./adjacency";
export type { PartitionPlan } from "./partition";

export interface MeshingInput {
  bbox: AABB;
  seeds: RefinementSeed[];
  octree?: Partial<OctreeOptions>;
  partitionCount?: number;
}

export interface MeshingResult {
  mesh: OctreeMesh;
  quality: QualityReport;
  adjacency: AdjacencyTensors;
  partition: PartitionPlan;
  summary: MeshingSummary;
}

export interface MeshingSummary {
  bbox: AABB;
  octree: {
    minDepth: number;
    maxDepth: number;
    leafCount: number;
    refinedFraction: number;
    boundaryLeaves: number;
    buildMs: number;
  };
  tets: {
    count: number;
    vertexCount: number;
    invertedTets: number;
    meanAspect: number;
    worstAspect: number;
    nonManifoldFaces: number;
    convergenceScore: number;
    aspectHist: number[];
  };
  adjacency: {
    edgeCount: number;
    meanDegree: number;
    maxDegree: number;
    warpStride: number;
  };
  partition: {
    partitionCount: number;
    sizes: number[];
    haloSizes: number[];
    edgeCut: number;
    imbalance: number;
    /** Flattened P×P comm matrix. */
    commMatrix: number[];
  };
  totalMs: number;
}

/** Translate STEP `desc.features` + bbox into refinement seeds. */
export function seedsFromFeatures(
  bbox: AABB,
  features: { kind: string; center?: number[]; radius?: number; weight?: number }[],
): RefinementSeed[] {
  const ext = Math.max(bbox.max[0] - bbox.min[0], bbox.max[1] - bbox.min[1], bbox.max[2] - bbox.min[2]);
  const fallbackR = ext * 0.08;
  const center = (): [number, number, number] => [
    (bbox.min[0] + bbox.max[0]) / 2,
    (bbox.min[1] + bbox.max[1]) / 2,
    (bbox.min[2] + bbox.max[2]) / 2,
  ];
  const map: Record<string, RefinementSeed["kind"]> = {
    hole: "hole", cylindrical: "hole",
    fillet: "fillet", toroidal: "fillet",
    sharp: "sharp", chamfer: "sharp", conical: "sharp",
    hotspot: "hotspot",
    overhang: "overhang",
    thin_wall: "thin_wall", thin: "thin_wall",
    contact: "contact",
  };
  return features
    .map((f) => {
      const kind = map[f.kind] ?? null;
      if (!kind) return null;
      const c = (f.center as [number, number, number] | undefined) ?? center();
      const r = f.radius ?? fallbackR;
      const w = f.weight ?? 0.6;
      return { kind, center: c, radius: r, weight: w } as RefinementSeed;
    })
    .filter((s): s is RefinementSeed => s !== null);
}

export function generateMesh(input: MeshingInput): MeshingResult {
  const t0 = Date.now();
  const mesh = buildOctreeMesh(input.bbox, input.seeds, input.octree);
  const quality = evaluateQuality(mesh);
  const adjacency = buildAdjacency(mesh);
  const partitionCount = Math.max(1, input.partitionCount ?? 8);
  const partition = partitionMesh(mesh, adjacency, partitionCount);

  const refinedFraction =
    mesh.leaves.length > 0
      ? mesh.leaves.filter((id) => mesh.nodes[id].depth >= (input.octree?.minDepth ?? 2) + 1).length /
        mesh.leaves.length
      : 0;

  const boundaryLeaves = Array.from(mesh.boundaryLeaf).reduce((a, b) => a + b, 0);

  const summary: MeshingSummary = {
    bbox: input.bbox,
    octree: {
      minDepth: mesh.options.minDepth,
      maxDepth: mesh.options.maxDepth,
      leafCount: mesh.leaves.length,
      refinedFraction,
      boundaryLeaves,
      buildMs: mesh.buildMs,
    },
    tets: {
      count: quality.tetCount,
      vertexCount: quality.vertexCount,
      invertedTets: quality.invertedTets,
      meanAspect: Number(quality.meanAspect.toFixed(3)),
      worstAspect: Number(quality.worstAspect.toFixed(3)),
      nonManifoldFaces: quality.nonManifoldFaces,
      convergenceScore: Number(quality.convergenceScore.toFixed(3)),
      aspectHist: quality.aspectHist,
    },
    adjacency: {
      edgeCount: adjacency.edgeStats.count,
      meanDegree: Number(adjacency.edgeStats.meanDegree.toFixed(2)),
      maxDegree: adjacency.edgeStats.maxDegree,
      warpStride: adjacency.warpStride,
    },
    partition: {
      partitionCount: partition.partitionCount,
      sizes: Array.from(partition.sizes),
      haloSizes: partition.halos.map((h) => h.length),
      edgeCut: partition.edgeCut,
      imbalance: Number(partition.imbalance.toFixed(3)),
      commMatrix: Array.from(partition.commMatrix),
    },
    totalMs: Date.now() - t0,
  };

  return { mesh, quality, adjacency, partition, summary };
}
