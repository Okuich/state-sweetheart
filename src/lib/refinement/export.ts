/**
 * Export the *refined local mesh* + refinement mask for downstream
 * Fabrication OS steps. Two formats:
 *
 *   - JSON: full structured dump (mesh, partition, mask, plan stats,
 *     pass metadata). Pure data — easiest to consume programmatically.
 *   - VTK (ASCII UnstructuredGrid): tetrahedral mesh + per-tet scalar
 *     fields (`partition`, `leaf`, `refined`, `coarsened`, `error`) so
 *     ParaView / downstream solvers visualize the mask directly.
 *
 * The "refinement mask" is per-leaf and per-tet:
 *   - leafRefined[li]   = 1 if leaf was split this pass
 *   - leafCoarsened[li] = 1 if leaf was selected for coarsening
 *   - tetRefined[t]     = leafRefined[tetLeaf[t]]
 *   - tetError[t]       = combined error indicator at the parent leaf
 */

import type { OctreeMesh } from "../meshing/octree";
import type { PartitionPlan } from "../meshing/partition";
import type { LeafErrorReport } from "./refine";
import type { AdaptivePassResult } from "./index";

export type RefinementExportFormat = "json" | "vtk";

export interface RefinementExportFile {
  filename: string;
  mimeType: string;
  content: string;
}

export interface RefinementMask {
  leafRefined: Uint8Array;
  leafCoarsened: Uint8Array;
  tetRefined: Uint8Array;
  tetError: Float32Array;
}

/**
 * Build the refinement mask aligned to the *base* mesh (the one whose
 * leaves were tagged for split/coarsen). Sized to base leaf/tet counts.
 */
export function buildRefinementMask(
  baseMesh: OctreeMesh,
  error: LeafErrorReport,
  splitLeaves: Uint32Array,
  coarsenLeaves: Uint32Array,
): RefinementMask {
  const L = baseMesh.leaves.length;
  const T = baseMesh.tets.length / 4;
  const leafRefined = new Uint8Array(L);
  const leafCoarsened = new Uint8Array(L);
  for (const li of splitLeaves) if (li < L) leafRefined[li] = 1;
  for (const li of coarsenLeaves) if (li < L) leafCoarsened[li] = 1;
  const tetRefined = new Uint8Array(T);
  const tetError = new Float32Array(T);
  for (let t = 0; t < T; t++) {
    const li = baseMesh.tetLeaf[t];
    tetRefined[t] = leafRefined[li] ?? 0;
    tetError[t] = error.combined[li] ?? 0;
  }
  return { leafRefined, leafCoarsened, tetRefined, tetError };
}

export interface RefinementExportInput {
  /** The result of the last adaptive pass (provides refined mesh + plan). */
  result: AdaptivePassResult;
  /** Base mesh that was passed *into* the pass (mask is sized to this). */
  baseMesh: OctreeMesh;
  /** Active partition (post-rebalance if distributed mode triggered one). */
  partition: PartitionPlan;
  /** Optional source label propagated into metadata. */
  source?: string;
}

export function exportRefinedMesh(
  input: RefinementExportInput,
  format: RefinementExportFormat,
): RefinementExportFile {
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const mask = buildRefinementMask(
    input.baseMesh,
    input.result.error,
    input.result.pass.plan.splitLeaves,
    input.result.pass.plan.coarsenLeaves,
  );
  if (format === "json") {
    return {
      filename: `refined-mesh-${stamp}.json`,
      mimeType: "application/json",
      content: toRefinedJSON(input, mask),
    };
  }
  return {
    filename: `refined-mesh-${stamp}.vtk`,
    mimeType: "model/vtk",
    content: toRefinedVTK(input, mask),
  };
}

function fmt(n: number): string {
  return Number.isFinite(n) ? n.toPrecision(7) : "0";
}

function toRefinedJSON(input: RefinementExportInput, mask: RefinementMask): string {
  const { result, baseMesh, partition, source } = input;
  const refined = result.pass.newMesh;
  const payload = {
    schema: "fabrication-os/refined-mesh@1",
    generatedAt: new Date().toISOString(),
    source: source ?? "physics-os/refinement",
    pass: {
      baseLeafCount: result.pass.baseLeafCount,
      refinedLeafCount: result.pass.refinedLeafCount,
      added: result.pass.added,
      removedSiblings: result.pass.removedSiblings,
      passMs: result.pass.passMs,
      fieldSource: result.fieldSource,
      shouldRepartition: result.shouldRepartition,
      planStats: result.pass.plan.stats,
    },
    bbox: refined.bbox,
    options: refined.options,
    refinedMesh: {
      vertices: Array.from(refined.vertices),
      tets: Array.from(refined.tets),
      tetLeaf: Array.from(refined.tetLeaf),
      boundaryLeaf: Array.from(refined.boundaryLeaf),
      leaves: refined.leaves.map((id) => {
        const n = refined.nodes[id];
        return { id, depth: n.depth, density: n.density, tag: n.tag, bbox: n.bbox };
      }),
    },
    baseMesh: {
      leafCount: baseMesh.leaves.length,
      tetCount: baseMesh.tets.length / 4,
    },
    refinementMask: {
      // Mask is base-mesh-aligned — Fabrication OS uses it to know which
      // regions were marked for higher-resolution toolpaths.
      leafRefined: Array.from(mask.leafRefined),
      leafCoarsened: Array.from(mask.leafCoarsened),
      tetRefined: Array.from(mask.tetRefined),
      tetError: Array.from(mask.tetError),
      splitLeafIds: Array.from(result.pass.plan.splitLeaves),
      coarsenLeafIds: Array.from(result.pass.plan.coarsenLeaves),
    },
    partition: {
      partitionCount: partition.partitionCount,
      tetPart: Array.from(partition.tetPart),
      sizes: Array.from(partition.sizes),
      imbalance: partition.imbalance,
    },
  };
  return JSON.stringify(payload, null, 2);
}

function toRefinedVTK(input: RefinementExportInput, mask: RefinementMask): string {
  const { result, partition } = input;
  const refined = result.pass.newMesh;
  const V = refined.vertices.length / 3;
  const T = refined.tets.length / 4;

  // Project the *base-mesh* mask onto the refined tets via tetLeaf identity
  // when the refined leaf survives, else mark as 1 (leaf was split into
  // children that all inherit the refined flag).
  const tetRefined = new Uint8Array(T);
  const tetError = new Float32Array(T);
  const baseLeafCount = result.pass.baseLeafCount;
  for (let t = 0; t < T; t++) {
    const li = refined.tetLeaf[t];
    if (li < baseLeafCount) {
      tetRefined[t] = mask.tetRefined[t < mask.tetRefined.length ? t : 0] ?? 0;
      tetError[t] = mask.tetError[t < mask.tetError.length ? t : 0] ?? 0;
    } else {
      // Leaf id beyond base count → newly created child of a split leaf.
      tetRefined[t] = 1;
      tetError[t] = 1;
    }
  }

  const lines: string[] = [];
  lines.push("# vtk DataFile Version 3.0");
  lines.push("Physics OS refined mesh + refinement mask");
  lines.push("ASCII");
  lines.push("DATASET UNSTRUCTURED_GRID");
  lines.push(`POINTS ${V} float`);
  for (let i = 0; i < V; i++) {
    lines.push(
      `${fmt(refined.vertices[i * 3])} ${fmt(refined.vertices[i * 3 + 1])} ${fmt(refined.vertices[i * 3 + 2])}`,
    );
  }
  lines.push(`CELLS ${T} ${T * 5}`);
  for (let t = 0; t < T; t++) {
    lines.push(
      `4 ${refined.tets[t * 4]} ${refined.tets[t * 4 + 1]} ${refined.tets[t * 4 + 2]} ${refined.tets[t * 4 + 3]}`,
    );
  }
  lines.push(`CELL_TYPES ${T}`);
  for (let t = 0; t < T; t++) lines.push("10");

  lines.push(`CELL_DATA ${T}`);
  lines.push("SCALARS partition int 1");
  lines.push("LOOKUP_TABLE default");
  for (let t = 0; t < T; t++) lines.push(String(partition.tetPart[t] ?? 0));
  lines.push("SCALARS leaf int 1");
  lines.push("LOOKUP_TABLE default");
  for (let t = 0; t < T; t++) lines.push(String(refined.tetLeaf[t]));
  lines.push("SCALARS refined int 1");
  lines.push("LOOKUP_TABLE default");
  for (let t = 0; t < T; t++) lines.push(String(tetRefined[t]));
  lines.push("SCALARS error float 1");
  lines.push("LOOKUP_TABLE default");
  for (let t = 0; t < T; t++) lines.push(fmt(tetError[t]));

  return lines.join("\n") + "\n";
}
