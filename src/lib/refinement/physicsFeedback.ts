/**
 * Physics OS → refinement feedback bridge.
 *
 * Replaces the synthetic field generator: real solver state (particle
 * positions, velocities, accelerations, contact events) is published from
 * `PhysicsCanvas` each frame and projected onto the refinement octree's
 * leaves to drive per-leaf error indicators.
 *
 * Field mapping (per leaf, normalized later by the refinement engine):
 *   stress       = mean ||a|| of particles in the leaf (deviatoric proxy)
 *   thermal      = kinetic energy density Σ ½m‖v‖² / leafVolume
 *   contact      = Σ penetration of contact events whose pair midpoint
 *                  falls in the leaf (or near it)
 *   deformation  = ||v - meanV|| (velocity divergence proxy)
 *   curvature    = local velocity curl magnitude across xy
 *   residual     = telemetry constraint_l2 / divergence_risk broadcast
 */

import type { OctreeMesh, Vec3 } from "../meshing/octree";
import type { PhysicsFields } from "./fields";

export interface PhysicsContactEvent {
  /** World coords (solver coordinate system). */
  pos: [number, number, number];
  penetration: number;
}

export interface PhysicsSnapshot {
  /** Wall-clock ms when published. */
  t: number;
  /** Solver source label, e.g. "PhysicsCanvas". */
  source: string;
  /** Particle count. */
  N: number;
  /** Spatial dimension of the solver (2 or 3). */
  D: 2 | 3;
  /** World-space extents the particles live in (px or m). */
  worldMin: [number, number, number];
  worldMax: [number, number, number];
  /** Length N*D positions (interleaved). */
  x: Float32Array | Float64Array;
  /** Length N*D velocities. */
  v: Float32Array | Float64Array;
  /** Length N*D accelerations (force / mass). Optional. */
  a?: Float32Array | Float64Array;
  /** Length N masses. Defaults to 1. */
  m?: Float32Array | Float64Array;
  /** Recent contact events. */
  contacts: PhysicsContactEvent[];
  /** Aggregate scalars from the telemetry channel. */
  scalars: {
    energyDriftPct?: number;
    constraintL2?: number;
    divergenceRisk?: number;
    velocityMax?: number;
    nanCount?: number;
  };
}

type Listener = (s: PhysicsSnapshot) => void;

class PhysicsFeedbackBus {
  private last: PhysicsSnapshot | null = null;
  private listeners = new Set<Listener>();
  private totalPublished = 0;

  publish(s: PhysicsSnapshot): void {
    this.last = s;
    this.totalPublished++;
    for (const l of this.listeners) {
      try { l(s); } catch { /* listener errors must not block */ }
    }
  }

  latest(): PhysicsSnapshot | null { return this.last; }

  ageMs(now: number = Date.now()): number {
    return this.last ? now - this.last.t : Infinity;
  }

  subscribe(l: Listener): () => void {
    this.listeners.add(l);
    return () => this.listeners.delete(l);
  }

  stats(): { totalPublished: number; listeners: number; lastT: number | null } {
    return {
      totalPublished: this.totalPublished,
      listeners: this.listeners.size,
      lastT: this.last ? this.last.t : null,
    };
  }

  clear(): void { this.last = null; }
}

const g = globalThis as unknown as { __physicsFeedbackBus?: PhysicsFeedbackBus };
if (!g.__physicsFeedbackBus) g.__physicsFeedbackBus = new PhysicsFeedbackBus();
/** Singleton bus shared across the worker. */
export const physicsFeedbackBus: PhysicsFeedbackBus = g.__physicsFeedbackBus;

function leafCenter(mesh: OctreeMesh, leafIdx: number): Vec3 {
  const id = mesh.leaves[leafIdx];
  const b = mesh.nodes[id].bbox;
  return [
    (b.min[0] + b.max[0]) / 2,
    (b.min[1] + b.max[1]) / 2,
    (b.min[2] + b.max[2]) / 2,
  ];
}

function leafVolume(mesh: OctreeMesh, leafIdx: number): number {
  const id = mesh.leaves[leafIdx];
  const b = mesh.nodes[id].bbox;
  return Math.max(1e-9, (b.max[0] - b.min[0]) * (b.max[1] - b.min[1]) * (b.max[2] - b.min[2]));
}

/**
 * Map a world-space point from the physics solver into the mesh bbox and
 * find the leaf containing that point. Returns -1 if no leaf is found.
 */
function locateLeaf(
  mesh: OctreeMesh,
  px: number, py: number, pz: number,
  snap: PhysicsSnapshot,
): number {
  const bb = mesh.bbox;
  const wMin = snap.worldMin, wMax = snap.worldMax;
  const u = (px - wMin[0]) / Math.max(1e-9, wMax[0] - wMin[0]);
  const v = (py - wMin[1]) / Math.max(1e-9, wMax[1] - wMin[1]);
  const w = snap.D === 3
    ? (pz - wMin[2]) / Math.max(1e-9, wMax[2] - wMin[2])
    : 0.5;
  const mx = bb.min[0] + Math.max(0, Math.min(1, u)) * (bb.max[0] - bb.min[0]);
  const my = bb.min[1] + Math.max(0, Math.min(1, v)) * (bb.max[1] - bb.min[1]);
  const mz = bb.min[2] + Math.max(0, Math.min(1, w)) * (bb.max[2] - bb.min[2]);
  // Brute-force point-in-bbox over leaves; fine for refinement-scale meshes (<1e5).
  for (let i = 0; i < mesh.leaves.length; i++) {
    const node = mesh.nodes[mesh.leaves[i]];
    const b = node.bbox;
    if (mx >= b.min[0] && mx <= b.max[0] &&
        my >= b.min[1] && my <= b.max[1] &&
        mz >= b.min[2] && mz <= b.max[2]) {
      return i;
    }
  }
  return -1;
}

/**
 * Project a Physics OS snapshot onto a refinement mesh and produce per-leaf
 * physics fields suitable for `computeLeafError`.
 *
 * Pure: no allocations escape, no globals consulted beyond inputs.
 */
export function projectFeedbackToFields(
  mesh: OctreeMesh,
  snap: PhysicsSnapshot,
): PhysicsFields {
  const N = mesh.leaves.length;
  const f: PhysicsFields = {
    stress: new Float32Array(N),
    thermal: new Float32Array(N),
    contact: new Float32Array(N),
    deformation: new Float32Array(N),
    curvature: new Float32Array(N),
    residual: new Float32Array(N),
  };
  const counts = new Uint32Array(N);
  const sumVx = new Float32Array(N);
  const sumVy = new Float32Array(N);
  const sumVz = new Float32Array(N);

  const D = snap.D;
  const Np = snap.N;

  // First pass: accumulate kinetic energy (thermal), |a| (stress), and
  // mean velocity per leaf.
  for (let i = 0; i < Np; i++) {
    const px = snap.x[i * D];
    const py = snap.x[i * D + 1];
    const pz = D === 3 ? snap.x[i * D + 2] : 0;
    const li = locateLeaf(mesh, px, py, pz, snap);
    if (li < 0) continue;

    const vx = snap.v[i * D];
    const vy = snap.v[i * D + 1];
    const vz = D === 3 ? snap.v[i * D + 2] : 0;
    const v2 = vx * vx + vy * vy + vz * vz;
    const mass = snap.m ? snap.m[i] : 1;

    const vol = leafVolume(mesh, li);
    f.thermal[li] += 0.5 * mass * v2 / vol;

    if (snap.a) {
      const ax = snap.a[i * D];
      const ay = snap.a[i * D + 1];
      const az = D === 3 ? snap.a[i * D + 2] : 0;
      f.stress[li] += Math.sqrt(ax * ax + ay * ay + az * az);
    } else {
      // Fall back: stress proxy = sqrt(v²) when no acceleration channel.
      f.stress[li] += Math.sqrt(v2);
    }

    sumVx[li] += vx;
    sumVy[li] += vy;
    sumVz[li] += vz;
    counts[li]++;
  }

  // Average accumulators by particle count where appropriate.
  for (let li = 0; li < N; li++) {
    const c = counts[li];
    if (c > 0) {
      f.stress[li] /= c;
      sumVx[li] /= c;
      sumVy[li] /= c;
      sumVz[li] /= c;
    }
  }

  // Second pass: deformation = mean |v - meanV| per leaf.
  for (let i = 0; i < Np; i++) {
    const px = snap.x[i * D];
    const py = snap.x[i * D + 1];
    const pz = D === 3 ? snap.x[i * D + 2] : 0;
    const li = locateLeaf(mesh, px, py, pz, snap);
    if (li < 0) continue;
    const dvx = snap.v[i * D] - sumVx[li];
    const dvy = snap.v[i * D + 1] - sumVy[li];
    const dvz = D === 3 ? snap.v[i * D + 2] - sumVz[li] : 0;
    f.deformation[li] += Math.sqrt(dvx * dvx + dvy * dvy + dvz * dvz);
  }
  for (let li = 0; li < N; li++) {
    const c = counts[li];
    if (c > 0) f.deformation[li] /= c;
  }

  // Curvature: discrete velocity curl across same-z neighbors using the
  // mean-velocity field per leaf. Approximation: ‖∂vy/∂x − ∂vx/∂y‖.
  // We sweep adjacent leaves in xy by indexing leaf centers into a sparse
  // grid bucketed by integer (col,row) at the average leaf size.
  const cellExt = leafExtX(mesh) || 1e-3;
  const cols = Math.max(1, Math.round((mesh.bbox.max[0] - mesh.bbox.min[0]) / cellExt));
  const rows = Math.max(1, Math.round((mesh.bbox.max[1] - mesh.bbox.min[1]) / cellExt));
  const grid = new Int32Array(cols * rows);
  grid.fill(-1);
  for (let li = 0; li < N; li++) {
    const c = leafCenter(mesh, li);
    const cx = Math.min(cols - 1, Math.max(0, Math.floor((c[0] - mesh.bbox.min[0]) / cellExt)));
    const cy = Math.min(rows - 1, Math.max(0, Math.floor((c[1] - mesh.bbox.min[1]) / cellExt)));
    grid[cy * cols + cx] = li;
  }
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const li = grid[r * cols + c];
      if (li < 0 || counts[li] === 0) continue;
      const liR = c + 1 < cols ? grid[r * cols + (c + 1)] : -1;
      const liU = r + 1 < rows ? grid[(r + 1) * cols + c] : -1;
      let dvy_dx = 0, dvx_dy = 0;
      if (liR >= 0 && counts[liR] > 0) dvy_dx = (sumVy[liR] - sumVy[li]) / cellExt;
      if (liU >= 0 && counts[liU] > 0) dvx_dy = (sumVx[liU] - sumVx[li]) / cellExt;
      f.curvature[li] = Math.abs(dvy_dx - dvx_dy);
    }
  }

  // Contacts: deposit each event's penetration into its containing leaf.
  for (const ev of snap.contacts) {
    const li = locateLeaf(mesh, ev.pos[0], ev.pos[1], ev.pos[2], snap);
    if (li < 0) continue;
    f.contact[li] += Math.max(0, ev.penetration);
  }

  // Residual: broadcast telemetry-derived solver-stability signal across
  // every leaf, biased toward leaves currently carrying particles.
  const constraint = snap.scalars.constraintL2 ?? 0;
  const divergence = snap.scalars.divergenceRisk ?? 0;
  const nan = (snap.scalars.nanCount ?? 0) > 0 ? 1 : 0;
  const baseResid = constraint + divergence + nan;
  if (baseResid > 0) {
    for (let li = 0; li < N; li++) {
      const local = counts[li] > 0 ? 1 : 0.25;
      f.residual[li] = baseResid * local;
    }
  }

  return f;
}

function leafExtX(mesh: OctreeMesh): number {
  if (mesh.leaves.length === 0) return 0;
  const id = mesh.leaves[0];
  const b = mesh.nodes[id].bbox;
  return b.max[0] - b.min[0];
}
