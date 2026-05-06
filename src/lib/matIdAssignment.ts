/**
 * matIdAssignment.ts
 * Per-element material-id assignment for a tetrahedral mesh.
 *
 * Each tetrahedron stores a MatID (uint32) indexing into a material
 * palette. The palette is a parallel array of {name, color, kind} so the
 * solver and visualizer can both look up a material in O(1).
 *
 * Operations:
 *  • paintByPredicate — assign MatID to every tet where pred(centroid) is true
 *  • paintByBoxRegion / paintBySphereRegion — common spatial selectors
 *  • paintByIndices — direct selection (e.g. picking)
 *  • histogram — per-material element counts (for the panel legend)
 *  • undo/redo via snapshot()/restore()
 */

export interface Tet {
  /** 4 vertex indices into a separate vertex array. */
  v: [number, number, number, number];
}

export interface TetMesh {
  vertices: Float32Array; // length V*3
  tets: Tet[];
  /** Length tets.length. matId[i] indexes paletteEntries. */
  matId: Uint32Array;
}

export interface MaterialPaletteEntry {
  id: number;
  name: string;
  /** CSS color string (any format the SVG/canvas accepts). */
  color: string;
  /** Constitutive kind tag from materialModel — kept loose to avoid coupling. */
  kind: string;
}

export const DEFAULT_PALETTE: MaterialPaletteEntry[] = [
  { id: 0, name: "Steel",     color: "#94a3b8", kind: "hookean" },
  { id: 1, name: "Rubber",    color: "#f97316", kind: "neo-hookean" },
  { id: 2, name: "Soft tissue", color: "#ec4899", kind: "viscoelastic" },
  { id: 3, name: "Aluminum",  color: "#38bdf8", kind: "plastic" },
  { id: 4, name: "Glass",     color: "#a3e635", kind: "fracture" },
];

// ── Centroid helper ──────────────────────────────────────────────────────

export function tetCentroid(mesh: TetMesh, i: number): [number, number, number] {
  const t = mesh.tets[i];
  const v = mesh.vertices;
  let cx = 0, cy = 0, cz = 0;
  for (const idx of t.v) {
    cx += v[idx * 3];
    cy += v[idx * 3 + 1];
    cz += v[idx * 3 + 2];
  }
  return [cx / 4, cy / 4, cz / 4];
}

// ── Painters ────────────────────────────────────────────────────────────

export function paintByPredicate(
  mesh: TetMesh,
  matId: number,
  pred: (centroid: [number, number, number], tetIdx: number) => boolean,
): number {
  let n = 0;
  for (let i = 0; i < mesh.tets.length; i++) {
    if (pred(tetCentroid(mesh, i), i)) {
      mesh.matId[i] = matId;
      n++;
    }
  }
  return n;
}

export function paintByBoxRegion(
  mesh: TetMesh, matId: number,
  min: [number, number, number], max: [number, number, number],
): number {
  return paintByPredicate(mesh, matId, ([x, y, z]) =>
    x >= min[0] && x <= max[0] &&
    y >= min[1] && y <= max[1] &&
    z >= min[2] && z <= max[2]);
}

export function paintBySphereRegion(
  mesh: TetMesh, matId: number,
  center: [number, number, number], radius: number,
): number {
  const r2 = radius * radius;
  return paintByPredicate(mesh, matId, ([x, y, z]) => {
    const dx = x - center[0], dy = y - center[1], dz = z - center[2];
    return dx * dx + dy * dy + dz * dz <= r2;
  });
}

export function paintByIndices(mesh: TetMesh, matId: number, idxs: ArrayLike<number>): number {
  let n = 0;
  for (let k = 0; k < idxs.length; k++) {
    const i = idxs[k];
    if (i >= 0 && i < mesh.tets.length) {
      mesh.matId[i] = matId;
      n++;
    }
  }
  return n;
}

// ── Histogram ───────────────────────────────────────────────────────────

export function histogram(mesh: TetMesh, palette: MaterialPaletteEntry[]): Map<number, number> {
  const out = new Map<number, number>();
  for (const p of palette) out.set(p.id, 0);
  for (let i = 0; i < mesh.matId.length; i++) {
    const m = mesh.matId[i];
    out.set(m, (out.get(m) ?? 0) + 1);
  }
  return out;
}

// ── Undo stack ──────────────────────────────────────────────────────────

export function snapshot(mesh: TetMesh): Uint32Array {
  return new Uint32Array(mesh.matId);
}

export function restore(mesh: TetMesh, snap: Uint32Array): void {
  if (snap.length !== mesh.matId.length) {
    throw new Error(`restore: snapshot length ${snap.length} ≠ mesh ${mesh.matId.length}`);
  }
  mesh.matId.set(snap);
}

// ── Demo mesh builder ───────────────────────────────────────────────────

/**
 * Build a regular tet-meshed unit-ish cube (NX×NY×NZ cells, 6 tets/cell).
 * Returns a mesh with all matId = 0.
 */
export function buildCubeMesh(NX: number, NY: number, NZ: number, size = 1): TetMesh {
  const verts: number[] = [];
  const idx = (ix: number, iy: number, iz: number) =>
    ix + (NX + 1) * (iy + (NY + 1) * iz);
  for (let iz = 0; iz <= NZ; iz++) {
    for (let iy = 0; iy <= NY; iy++) {
      for (let ix = 0; ix <= NX; ix++) {
        verts.push((ix / NX) * size, (iy / NY) * size, (iz / NZ) * size);
      }
    }
  }
  const tets: Tet[] = [];
  // 6-tet decomposition of a hex.
  const hex = [
    [0, 1, 3, 4], [1, 2, 3, 6], [1, 3, 4, 6],
    [3, 4, 6, 7], [1, 4, 5, 6], [4, 5, 6, 7],
  ];
  for (let iz = 0; iz < NZ; iz++) {
    for (let iy = 0; iy < NY; iy++) {
      for (let ix = 0; ix < NX; ix++) {
        const corners = [
          idx(ix,     iy,     iz),     idx(ix + 1, iy,     iz),
          idx(ix + 1, iy + 1, iz),     idx(ix,     iy + 1, iz),
          idx(ix,     iy,     iz + 1), idx(ix + 1, iy,     iz + 1),
          idx(ix + 1, iy + 1, iz + 1), idx(ix,     iy + 1, iz + 1),
        ];
        for (const t of hex) {
          tets.push({ v: [corners[t[0]], corners[t[1]], corners[t[2]], corners[t[3]]] });
        }
      }
    }
  }
  return {
    vertices: Float32Array.from(verts),
    tets,
    matId: new Uint32Array(tets.length),
  };
}
