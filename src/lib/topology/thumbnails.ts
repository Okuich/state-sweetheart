/**
 * Topology thumbnail renderer.
 *
 * Rasterizes small canvas previews of a TopologyResult for embedding into
 * the PDF report. We render four orthographic views:
 *   - XY projection colored by feature class
 *   - XZ projection colored by feature class
 *   - YZ projection colored by feature class
 *   - XY projection colored by partition owner
 *
 * Each leaf is drawn as a filled square at its projected center, sized by
 * its equivalent radius. Pure 2D canvas — no WebGL required.
 */

import type { TopologyResult } from "./index";
import type { FeatureClass } from "./types";

const FEATURE_COLORS: Record<FeatureClass, string> = {
  bulk: "#cbd5e1",
  boundary: "#94a3b8",
  thin_wall: "#f59e0b",
  overhang: "#fb7185",
  cavity: "#a855f7",
  stress_concentrator: "#ef4444",
  thermal_bottleneck: "#06b6d4",
  symmetry_seed: "#22c55e",
};

// Distinct partition palette (cycled).
const PARTITION_COLORS = [
  "#3b82f6", "#ef4444", "#22c55e", "#f59e0b",
  "#a855f7", "#06b6d4", "#ec4899", "#84cc16",
  "#f97316", "#14b8a6", "#8b5cf6", "#eab308",
];

export interface TopologyThumbnail {
  dataUrl: string;
  title: string;
}

export interface ThumbnailOptions {
  /** Pixel dimensions per thumbnail (square). */
  size?: number;
  /** DPR multiplier for sharper PDF embedding. */
  scale?: number;
  /** Background color (canvas fill). */
  background?: string;
}

type Axis = 0 | 1 | 2;

function project(center: [number, number, number], hAxis: Axis, vAxis: Axis): [number, number] {
  return [center[hAxis], center[vAxis]];
}

function renderProjection(
  r: TopologyResult,
  hAxis: Axis,
  vAxis: Axis,
  colorFn: (nodeIdx: number) => string,
  opts: Required<ThumbnailOptions>,
): string | null {
  if (typeof document === "undefined") return null;
  const px = opts.size * opts.scale;
  const canvas = document.createElement("canvas");
  canvas.width = px;
  canvas.height = px;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.fillStyle = opts.background;
  ctx.fillRect(0, 0, px, px);

  const nodes = r.graph.nodes;
  if (!nodes.length) return canvas.toDataURL("image/png");

  // Bounds across the projected plane.
  let minH = Infinity, maxH = -Infinity, minV = Infinity, maxV = -Infinity;
  let maxR = 0;
  for (const n of nodes) {
    const [h, v] = project(n.center, hAxis, vAxis);
    if (h < minH) minH = h;
    if (h > maxH) maxH = h;
    if (v < minV) minV = v;
    if (v > maxV) maxV = v;
    if (n.radius > maxR) maxR = n.radius;
  }
  const padW = (maxH - minH) || 1;
  const padH = (maxV - minV) || 1;
  const span = Math.max(padW, padH);
  const margin = px * 0.06;
  const usable = px - margin * 2;
  const scale = usable / span;
  const cx = (minH + maxH) / 2;
  const cy = (minV + maxV) / 2;

  // Sort by depth on the third axis so closer leaves render last.
  const dAxis = (3 - hAxis - vAxis) as Axis;
  const order = nodes.map((_, i) => i).sort((a, b) => nodes[a].center[dAxis] - nodes[b].center[dAxis]);

  for (const i of order) {
    const n = nodes[i];
    const [h, v] = project(n.center, hAxis, vAxis);
    const x = px / 2 + (h - cx) * scale;
    // Flip Y so positive points up like a CAD view.
    const y = px / 2 - (v - cy) * scale;
    const s = Math.max(1.2 * opts.scale, n.radius * scale * 1.4);
    ctx.fillStyle = colorFn(i);
    ctx.globalAlpha = n.feature === "bulk" ? 0.55 : 0.92;
    ctx.fillRect(x - s / 2, y - s / 2, s, s);
  }
  ctx.globalAlpha = 1;
  // Frame.
  ctx.strokeStyle = "rgba(15,23,42,0.45)";
  ctx.lineWidth = opts.scale;
  ctx.strokeRect(0.5, 0.5, px - 1, px - 1);

  return canvas.toDataURL("image/png");
}

export function renderTopologyThumbnails(
  r: TopologyResult,
  options: ThumbnailOptions = {},
): TopologyThumbnail[] {
  const opts: Required<ThumbnailOptions> = {
    size: options.size ?? 220,
    scale: options.scale ?? 2,
    background: options.background ?? "#ffffff",
  };
  const featureColor = (i: number) => FEATURE_COLORS[r.graph.nodes[i].feature] ?? "#cbd5e1";
  const partColor = (i: number) => {
    const owner = r.partition.owners[i];
    if (owner < 0) return "#e5e7eb";
    return PARTITION_COLORS[owner % PARTITION_COLORS.length];
  };

  const out: TopologyThumbnail[] = [];
  const xy = renderProjection(r, 0, 1, featureColor, opts);
  const xz = renderProjection(r, 0, 2, featureColor, opts);
  const yz = renderProjection(r, 1, 2, featureColor, opts);
  const part = renderProjection(r, 0, 1, partColor, opts);
  if (xy) out.push({ dataUrl: xy, title: "XY · features" });
  if (xz) out.push({ dataUrl: xz, title: "XZ · features" });
  if (yz) out.push({ dataUrl: yz, title: "YZ · features" });
  if (part) out.push({ dataUrl: part, title: `XY · partitions (P=${r.partition.partitionCount})` });
  return out;
}

export const FEATURE_THUMB_LEGEND: { label: string; color: string }[] = (
  Object.entries(FEATURE_COLORS) as [FeatureClass, string][]
).map(([k, v]) => ({ label: k.replace(/_/g, " "), color: v }));
