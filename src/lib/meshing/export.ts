/**
 * Mesh exporters: VTK (ASCII UnstructuredGrid), OBJ (triangulated tet
 * surfaces), and JSON (full structured dump). Pure functions so they run
 * identically in the browser and inside server functions.
 */

import type { MeshingResult } from "./index";

export type ExportFormat = "vtk" | "obj" | "json";

export interface ExportFile {
  filename: string;
  mimeType: string;
  content: string;
}

const TET_FACES: [number, number, number][] = [
  [0, 1, 2],
  [0, 1, 3],
  [0, 2, 3],
  [1, 2, 3],
];

export function exportMesh(result: MeshingResult, format: ExportFormat): ExportFile {
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  switch (format) {
    case "vtk":
      return {
        filename: `mesh-${stamp}.vtk`,
        mimeType: "model/vtk",
        content: toVTK(result),
      };
    case "obj":
      return {
        filename: `mesh-${stamp}.obj`,
        mimeType: "model/obj",
        content: toOBJ(result),
      };
    case "json":
      return {
        filename: `mesh-${stamp}.json`,
        mimeType: "application/json",
        content: toJSON(result),
      };
  }
}

function toVTK(r: MeshingResult): string {
  const { mesh, partition } = r;
  const V = mesh.vertices.length / 3;
  const T = mesh.tets.length / 4;
  const lines: string[] = [];
  lines.push("# vtk DataFile Version 3.0");
  lines.push("Physics OS adaptive tetrahedral mesh");
  lines.push("ASCII");
  lines.push("DATASET UNSTRUCTURED_GRID");
  lines.push(`POINTS ${V} float`);
  for (let i = 0; i < V; i++) {
    lines.push(
      `${fmt(mesh.vertices[i * 3])} ${fmt(mesh.vertices[i * 3 + 1])} ${fmt(mesh.vertices[i * 3 + 2])}`,
    );
  }
  lines.push(`CELLS ${T} ${T * 5}`);
  for (let t = 0; t < T; t++) {
    lines.push(
      `4 ${mesh.tets[t * 4]} ${mesh.tets[t * 4 + 1]} ${mesh.tets[t * 4 + 2]} ${mesh.tets[t * 4 + 3]}`,
    );
  }
  lines.push(`CELL_TYPES ${T}`);
  for (let t = 0; t < T; t++) lines.push("10"); // VTK_TETRA

  // Cell data: partition id + parent leaf
  lines.push(`CELL_DATA ${T}`);
  lines.push("SCALARS partition int 1");
  lines.push("LOOKUP_TABLE default");
  for (let t = 0; t < T; t++) lines.push(String(partition.tetPart[t] ?? 0));
  lines.push("SCALARS leaf int 1");
  lines.push("LOOKUP_TABLE default");
  for (let t = 0; t < T; t++) lines.push(String(mesh.tetLeaf[t]));

  return lines.join("\n") + "\n";
}

function toOBJ(r: MeshingResult): string {
  const { mesh } = r;
  const V = mesh.vertices.length / 3;
  const T = mesh.tets.length / 4;
  const lines: string[] = [];
  lines.push("# Physics OS tetrahedral mesh");
  lines.push(`# vertices=${V} tets=${T}`);
  for (let i = 0; i < V; i++) {
    lines.push(
      `v ${fmt(mesh.vertices[i * 3])} ${fmt(mesh.vertices[i * 3 + 1])} ${fmt(mesh.vertices[i * 3 + 2])}`,
    );
  }
  // OBJ is 1-indexed. Emit each tet as 4 triangle faces.
  for (let t = 0; t < T; t++) {
    const v0 = mesh.tets[t * 4] + 1;
    const v1 = mesh.tets[t * 4 + 1] + 1;
    const v2 = mesh.tets[t * 4 + 2] + 1;
    const v3 = mesh.tets[t * 4 + 3] + 1;
    const corners = [v0, v1, v2, v3];
    for (const f of TET_FACES) {
      lines.push(`f ${corners[f[0]]} ${corners[f[1]]} ${corners[f[2]]}`);
    }
  }
  return lines.join("\n") + "\n";
}

function toJSON(r: MeshingResult): string {
  const { mesh, partition, adjacency, summary } = r;
  const payload = {
    bbox: mesh.bbox,
    options: mesh.options,
    summary,
    vertices: Array.from(mesh.vertices),
    tets: Array.from(mesh.tets),
    tetLeaf: Array.from(mesh.tetLeaf),
    boundaryLeaf: Array.from(mesh.boundaryLeaf),
    leaves: mesh.leaves.map((id) => {
      const n = mesh.nodes[id];
      return { id, depth: n.depth, density: n.density, tag: n.tag, bbox: n.bbox };
    }),
    partition: {
      partitionCount: partition.partitionCount,
      tetPart: Array.from(partition.tetPart),
      sizes: Array.from(partition.sizes),
      halos: partition.halos.map((h) => Array.from(h)),
      commMatrix: Array.from(partition.commMatrix),
      edgeCut: partition.edgeCut,
      imbalance: partition.imbalance,
    },
    adjacency: {
      edgeStats: adjacency.edgeStats,
      warpStride: adjacency.warpStride,
    },
  };
  return JSON.stringify(payload, null, 2);
}

function fmt(n: number): string {
  // Compact but precise enough for meshing.
  return Number.isFinite(n) ? n.toPrecision(7) : "0";
}
