/**
 * One-click report export for Topology Intelligence results.
 *
 *   buildReportJSON(result)  →  serializable JSON snapshot
 *   buildReportPDF(result)   →  jsPDF Blob with the same data
 *   downloadReport(result, fmt) → triggers browser download
 */

import { jsPDF } from "jspdf";
import { FEATURE_LABELS, type FeatureClass } from "./types";
import type { TopologyResult } from "./index";
import { renderTopologyThumbnails, FEATURE_THUMB_LEGEND } from "./thumbnails";

const FEATURE_ORDER: FeatureClass[] = [
  "bulk", "boundary", "thin_wall", "overhang",
  "cavity", "stress_concentrator", "thermal_bottleneck", "symmetry_seed",
];

export interface TopologyReport {
  generatedAt: string;
  pipelineMs: number;
  /** Sections that were included in this snapshot (omitted = all). */
  sections?: ReportSections;
  graph?: {
    nodes: number;
    edges: number;
    meanValence: number;
    buildMs: number;
  };
  features?: {
    counts: Record<FeatureClass, number>;
    symmetryScore: number;
    minWallThickness: number;
    avgThinWallThickness: number;
  };
  manufacturability?: {
    feasibility: number;
    machiningAccess: number;
    supportFraction: number;
    thermalDistortionRisk: number;
    assemblyComplexity: number;
    drivers: {
      overhangPenalty: number;
      cavityPenalty: number;
      thinWallPenalty: number;
      stressPenalty: number;
      thermalPenalty: number;
    };
  };
  priors?: {
    timestepScale: number;
    damping: number;
    contactStiffness: number;
    refinementHints: Record<FeatureClass, number>;
  };
  partition?: {
    partitionCount: number;
    edgeCut: number;
    imbalance: number;
    resident: number[];
    halos: number[];
    commMatrix: number[];
  };
  embedding?: {
    dim: number;
    slices: { curvature: [number, number]; features: [number, number]; structural: [number, number]; manuf: [number, number] };
    vector: number[];
  };
}

export interface ReportSections {
  graph?: boolean;
  features?: boolean;
  manufacturability?: boolean;
  partition?: boolean;
  embedding?: boolean;
  /** PDF-only: include the auto-generated topology thumbnail page. */
  thumbnails?: boolean;
}

export const ALL_SECTIONS: Required<ReportSections> = {
  graph: true,
  features: true,
  manufacturability: true,
  partition: true,
  embedding: true,
  thumbnails: true,
};

function resolveSections(s?: ReportSections): Required<ReportSections> {
  return { ...ALL_SECTIONS, ...(s ?? {}) };
}

export function buildReportJSON(r: TopologyResult, sections?: ReportSections): TopologyReport {
  const sec = resolveSections(sections);
  const meanValence = (r.graph.edges.length * 2) / Math.max(1, r.graph.nodes.length);
  const out: TopologyReport = {
    generatedAt: new Date().toISOString(),
    pipelineMs: r.totalMs,
    sections: sec,
  };
  if (sec.graph) {
    out.graph = {
      nodes: r.graph.nodes.length,
      edges: r.graph.edges.length,
      meanValence,
      buildMs: r.graph.buildMs,
    };
  }
  if (sec.features) {
    out.features = {
      counts: r.features.counts,
      symmetryScore: r.features.symmetryScore,
      minWallThickness: r.features.minWallThickness,
      avgThinWallThickness: r.features.avgThinWallThickness,
    };
  }
  if (sec.manufacturability) {
    out.manufacturability = {
      feasibility: r.manufacturability.feasibility,
      machiningAccess: r.manufacturability.machiningAccess,
      supportFraction: r.manufacturability.supportFraction,
      thermalDistortionRisk: r.manufacturability.thermalDistortionRisk,
      assemblyComplexity: r.manufacturability.assemblyComplexity,
      drivers: r.manufacturability.drivers,
    };
    out.priors = {
      timestepScale: r.priors.timestepScale,
      damping: r.priors.damping,
      contactStiffness: r.priors.contactStiffness,
      refinementHints: r.priors.refinementHints,
    };
  }
  if (sec.partition) {
    out.partition = {
      partitionCount: r.partition.partitionCount,
      edgeCut: r.partition.edgeCut,
      imbalance: r.partition.imbalance,
      resident: r.partition.resident.map((p) => p.length),
      halos: r.partition.halos.map((p) => p.length),
      commMatrix: Array.from(r.partition.commMatrix),
    };
  }
  if (sec.embedding) {
    out.embedding = {
      dim: r.embedding.vector.length,
      slices: r.embedding.slices,
      vector: Array.from(r.embedding.vector),
    };
  }
  return out;
}

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  const v = parseInt(h.length === 3 ? h.split("").map((c) => c + c).join("") : h, 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

export function buildReportPDF(r: TopologyResult, label?: string, sections?: ReportSections): Blob {
  const sec = resolveSections(sections);
  const rep = buildReportJSON(r, sec);
  const doc = new jsPDF({ unit: "pt", format: "a4" });
  const W = doc.internal.pageSize.getWidth();
  const H = doc.internal.pageSize.getHeight();
  const M = 40;
  let y = M;

  const ensure = (h: number) => {
    if (y + h > H - M) { doc.addPage(); y = M; }
  };
  const h1 = (t: string) => {
    ensure(28);
    doc.setFont("helvetica", "bold"); doc.setFontSize(16);
    doc.text(t, M, y); y += 22;
  };
  const h2 = (t: string) => {
    ensure(20);
    doc.setFont("helvetica", "bold"); doc.setFontSize(12);
    doc.text(t, M, y); y += 16;
  };
  const kv = (rows: [string, string][]) => {
    doc.setFont("helvetica", "normal"); doc.setFontSize(10);
    for (const [k, v] of rows) {
      ensure(14);
      doc.setTextColor(110); doc.text(k, M, y);
      doc.setTextColor(20); doc.text(v, M + 200, y);
      y += 13;
    }
    doc.setTextColor(20);
  };
  const bar = (label: string, value: number, max = 1) => {
    ensure(14);
    doc.setFont("helvetica", "normal"); doc.setFontSize(9);
    doc.setTextColor(80); doc.text(label, M, y);
    const bx = M + 160, bw = W - M - bx - 60;
    doc.setDrawColor(220); doc.setFillColor(235, 235, 235);
    doc.rect(bx, y - 8, bw, 9, "FD");
    const pct = Math.max(0, Math.min(1, value / max));
    doc.setFillColor(60, 110, 200);
    doc.rect(bx, y - 8, bw * pct, 9, "F");
    doc.setTextColor(20);
    doc.text(value.toFixed(3), bx + bw + 8, y);
    y += 13;
  };

  // Header
  doc.setFont("helvetica", "bold"); doc.setFontSize(20);
  doc.text("Topology Intelligence Report", M, y); y += 24;
  doc.setFont("helvetica", "normal"); doc.setFontSize(9); doc.setTextColor(110);
  doc.text(`${label ?? "preset"} · generated ${rep.generatedAt}`, M, y); y += 18;
  doc.setTextColor(20);

  // Auto-generated thumbnails of the topology (XY/XZ/YZ + partition view).
  const thumbs = renderTopologyThumbnails(r, { size: 220, scale: 2 });
  if (thumbs.length) {
    h1("Topology views");
    const cols = 2;
    const gap = 12;
    const tw = (W - M * 2 - gap * (cols - 1)) / cols;
    const th = tw; // square renders
    const captionH = 14;
    for (let i = 0; i < thumbs.length; i += cols) {
      ensure(th + captionH + 4);
      const rowY = y;
      for (let c = 0; c < cols && i + c < thumbs.length; c++) {
        const t = thumbs[i + c];
        const x = M + c * (tw + gap);
        try {
          doc.addImage(t.dataUrl, "PNG", x, rowY, tw, th);
        } catch {
          // If the canvas/dataUrl failed, fall back to a placeholder rect.
          doc.setDrawColor(200); doc.rect(x, rowY, tw, th);
        }
        doc.setFont("helvetica", "normal"); doc.setFontSize(8); doc.setTextColor(110);
        doc.text(t.title, x + 4, rowY + th + 10);
        doc.setTextColor(20);
      }
      y = rowY + th + captionH;
    }
    // Compact legend for the feature-colored views.
    ensure(18);
    doc.setFont("helvetica", "normal"); doc.setFontSize(7); doc.setTextColor(90);
    let lx = M;
    const swatch = 7;
    for (const item of FEATURE_THUMB_LEGEND) {
      const text = item.label;
      const tWidth = doc.getTextWidth(text);
      if (lx + swatch + 3 + tWidth + 8 > W - M) {
        y += 10;
        ensure(10);
        lx = M;
      }
      const [rr, gg, bb] = hexToRgb(item.color);
      doc.setFillColor(rr, gg, bb);
      doc.rect(lx, y - swatch + 1, swatch, swatch, "F");
      doc.text(text, lx + swatch + 3, y);
      lx += swatch + 3 + tWidth + 8;
    }
    doc.setTextColor(20);
    y += 10;
  }

  h1("Topology graph");
  kv([
    ["nodes", `${rep.graph.nodes}`],
    ["edges", `${rep.graph.edges}`],
    ["mean valence", rep.graph.meanValence.toFixed(2)],
    ["graph build", `${rep.graph.buildMs} ms`],
    ["pipeline total", `${rep.pipelineMs} ms`],
    ["symmetry", `${(rep.features.symmetryScore * 100).toFixed(1)}%`],
    ["min wall thickness", rep.features.minWallThickness.toFixed(3)],
    ["avg thin-wall thickness", rep.features.avgThinWallThickness.toFixed(3)],
  ]);

  h1("Feature classification");
  for (const c of FEATURE_ORDER) bar(FEATURE_LABELS[c] as string, rep.features.counts[c], Math.max(1, rep.graph.nodes));

  h1("Manufacturability");
  const m = rep.manufacturability;
  bar("feasibility", m.feasibility);
  bar("machining access", m.machiningAccess);
  bar("support fraction", m.supportFraction);
  bar("thermal distortion risk", m.thermalDistortionRisk);
  bar("assembly complexity", m.assemblyComplexity);

  h2("Drivers");
  bar("overhang penalty", m.drivers.overhangPenalty);
  bar("cavity penalty", m.drivers.cavityPenalty);
  bar("thin-wall penalty", m.drivers.thinWallPenalty);
  bar("stress penalty", m.drivers.stressPenalty);
  bar("thermal penalty", m.drivers.thermalPenalty);

  h1("Physics priors");
  kv([
    ["timestep scale", rep.priors.timestepScale.toFixed(3)],
    ["damping", rep.priors.damping.toFixed(3)],
    ["contact stiffness", rep.priors.contactStiffness.toFixed(3)],
  ]);
  h2("Refinement hints");
  for (const c of FEATURE_ORDER) bar(FEATURE_LABELS[c], rep.priors.refinementHints[c], 2);

  h1(`Partitions · P=${rep.partition.partitionCount}`);
  kv([
    ["edge cut", `${rep.partition.edgeCut}`],
    ["imbalance", `${(rep.partition.imbalance * 100).toFixed(1)}%`],
  ]);
  h2("Resident / halo per partition");
  doc.setFont("helvetica", "normal"); doc.setFontSize(9);
  for (let i = 0; i < rep.partition.partitionCount; i++) {
    ensure(12);
    doc.text(`P${i}: resident ${rep.partition.resident[i]} · halo ${rep.partition.halos[i]}`, M, y);
    y += 12;
  }

  h2("Communication matrix (rows = receiver)");
  const P = rep.partition.partitionCount;
  const maxFlow = Math.max(1, ...rep.partition.commMatrix);
  const labelW = 30;        // left "P##" gutter
  const headerH = 12;       // top "P##" header strip
  const availW = W - M * 2 - labelW;
  const availH = H - M - y - 4; // remaining height on current page
  // Pick a target cell size that keeps the whole matrix legible. Cells
  // shrink with P, but never below 4pt (pure heatmap, no inline text).
  const idealCell = Math.min(36, Math.max(4, Math.floor(720 / Math.max(8, P))));
  const cellW = Math.max(4, Math.min(idealCell, Math.floor(availW)));
  const cellH = cellW; // square cells regardless of page
  // How many cols fit across one page; how many rows fit per page block.
  const colsPerPage = Math.max(1, Math.min(P, Math.floor(availW / cellW)));
  const rowsFirstPage = Math.max(1, Math.floor((availH - headerH) / cellH));
  const rowsFullPage = Math.max(1, Math.floor((H - M * 2 - headerH) / cellH));
  const showText = cellW >= 18 && cellH >= 14;
  const numCol = (n: number) => `P${n}`;

  for (let c0 = 0; c0 < P; c0 += colsPerPage) {
    const cN = Math.min(P, c0 + colsPerPage);
    let r0 = 0;
    let firstBlockOnThisColRange = true;
    while (r0 < P) {
      // Decide capacity: first block reuses leftover space on the current
      // page; subsequent blocks for the same column-range start fresh.
      if (!firstBlockOnThisColRange || c0 > 0) {
        doc.addPage();
        y = M;
      }
      const cap = firstBlockOnThisColRange && c0 === 0
        ? rowsFirstPage
        : rowsFullPage;
      const rN = Math.min(P, r0 + cap);
      // Sub-block caption when matrix paginates.
      if (P > colsPerPage || rN - r0 < P) {
        doc.setFont("helvetica", "italic"); doc.setFontSize(8); doc.setTextColor(110);
        doc.text(`cols ${numCol(c0)}–${numCol(cN - 1)} · rows ${numCol(r0)}–${numCol(rN - 1)} of P=${P}`, M, y);
        y += 10;
        doc.setTextColor(20);
      }
      // Column header.
      doc.setFont("helvetica", "normal"); doc.setFontSize(Math.min(8, Math.max(5, cellW * 0.45)));
      for (let c = c0; c < cN; c++) {
        const cx = M + labelW + (c - c0) * cellW + cellW / 2;
        doc.text(numCol(c), cx, y, { align: "center" } as { align: "center" });
      }
      y += headerH - 2;
      // Body rows.
      for (let r2 = r0; r2 < rN; r2++) {
        doc.setFontSize(Math.min(8, Math.max(5, cellH * 0.45)));
        doc.text(numCol(r2), M, y + cellH - 4);
        for (let c = c0; c < cN; c++) {
          const v = rep.partition.commMatrix[r2 * P + c];
          const t = v / maxFlow;
          doc.setFillColor(255 - Math.round(t * 195), 255 - Math.round(t * 145), 255 - Math.round(t * 55));
          doc.rect(M + labelW + (c - c0) * cellW, y, cellW - 1, cellH - 1, "F");
          if (showText && v > 0) {
            doc.setTextColor(t > 0.55 ? 255 : 30);
            doc.text(`${v}`, M + labelW + (c - c0) * cellW + cellW / 2, y + cellH - 5, { align: "center" } as { align: "center" });
          }
        }
        y += cellH;
      }
      doc.setTextColor(20);
      y += 6;
      r0 = rN;
      firstBlockOnThisColRange = false;
    }
  }
  y += 2;

  h1(`Structural embedding (${rep.embedding.dim}-d)`);
  doc.setFontSize(8); doc.setTextColor(110);
  const sl = rep.embedding.slices;
  doc.text(`slices: curvature [${sl.curvature[0]}–${sl.curvature[1]}] · features [${sl.features[0]}–${sl.features[1]}] · structural [${sl.structural[0]}–${sl.structural[1]}] · manuf [${sl.manuf[0]}–${sl.manuf[1]}]`, M, y);
  y += 12; doc.setTextColor(20);

  ensure(40);
  const stripW = W - M * 2;
  const cw = stripW / rep.embedding.dim;
  const maxAbs = Math.max(1e-6, ...rep.embedding.vector.map((v) => Math.abs(v)));
  for (let i = 0; i < rep.embedding.dim; i++) {
    const v = rep.embedding.vector[i];
    const t = Math.abs(v) / maxAbs;
    if (v >= 0) doc.setFillColor(40, 90, 200, );
    else doc.setFillColor(200, 70, 40);
    const shade = 40 + Math.round(t * 200);
    if (v >= 0) doc.setFillColor(255 - shade, 255 - shade, 255);
    else doc.setFillColor(255, 255 - shade, 255 - shade);
    doc.rect(M + i * cw, y, cw, 22, "F");
  }
  y += 28;

  // Raw vector
  doc.setFont("courier", "normal"); doc.setFontSize(7); doc.setTextColor(70);
  const raw = rep.embedding.vector.map((v) => v.toFixed(3)).join(", ");
  const lines = doc.splitTextToSize(raw, W - M * 2) as string[];
  for (const ln of lines) { ensure(9); doc.text(ln, M, y); y += 9; }

  return doc.output("blob");
}

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function downloadReport(r: TopologyResult, fmt: "pdf" | "json", label?: string) {
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const slug = (label ?? "topology").replace(/[^a-z0-9]+/gi, "-").toLowerCase();
  if (fmt === "json") {
    const blob = new Blob([JSON.stringify(buildReportJSON(r), null, 2)], { type: "application/json" });
    downloadBlob(blob, `${slug}-report-${stamp}.json`);
  } else {
    downloadBlob(buildReportPDF(r, label), `${slug}-report-${stamp}.pdf`);
  }
}

/**
 * Validate that an unknown value matches the TopologyReport shape we
 * emit from buildReportJSON. Throws a descriptive Error on the first
 * problem so the caller can surface it to the user.
 */
export function parseTopologyReport(input: unknown): TopologyReport {
  if (!input || typeof input !== "object") throw new Error("not a JSON object");
  const o = input as Record<string, unknown>;
  const need = (k: string) => {
    if (!(k in o)) throw new Error(`missing field "${k}"`);
  };
  for (const k of ["graph", "features", "manufacturability", "priors", "partition", "embedding"]) need(k);
  const part = o.partition as Record<string, unknown>;
  if (typeof part.partitionCount !== "number") throw new Error("partition.partitionCount missing");
  if (!Array.isArray(part.commMatrix)) throw new Error("partition.commMatrix missing");
  const expected = (part.partitionCount as number) ** 2;
  if ((part.commMatrix as unknown[]).length !== expected) {
    throw new Error(`partition.commMatrix length ${(part.commMatrix as unknown[]).length} ≠ P² (${expected})`);
  }
  const emb = o.embedding as Record<string, unknown>;
  if (!Array.isArray(emb.vector)) throw new Error("embedding.vector missing");
  // At this point the structural checks pass; trust the snapshot.
  return input as TopologyReport;
}

/**
 * Reconstruct a TopologyResult-shaped object from a previously-exported
 * report snapshot, sufficient to re-render the dashboard panels.
 *
 * The original graph nodes/edges, owner ids and resident lists are not
 * round-trippable from JSON (the JSON stores only counts and aggregates),
 * so the rehydrated result uses placeholder graph nodes whose .length
 * matches the snapshot. Panels that read aggregate counts/sizes work
 * unchanged; features that require live graph topology (GPU traversal
 * bench, retrieval) should be gated by an `imported` flag in the UI.
 */
export function rehydrateFromReport(rep: TopologyReport): TopologyResult {
  const N = rep.graph.nodes;
  const E = rep.graph.edges;
  const P = rep.partition.partitionCount;

  // Placeholder node objects — shape-correct, content irrelevant for the
  // panels (which only read .length).
  const nodes = new Array(N).fill(null).map((_, i) => ({
    leaf: i,
    center: [0, 0, 0] as [number, number, number],
    radius: 0,
    density: 0,
    tag: "bulk",
    feature: "bulk" as const,
    wallThickness: 0,
    curvature: 0,
    boundary: false,
    downward: false,
  }));
  const edges = new Array(E).fill(null).map(() => ({ a: 0, b: 0, shared: 0, axis: 0 as const }));

  // Placeholder resident lists with the right cardinality per partition.
  const resident: number[][] = new Array(P).fill(null).map((_, p) => {
    const len = rep.partition.resident[p] ?? 0;
    const arr = new Array(len);
    for (let i = 0; i < len; i++) arr[i] = i;
    return arr;
  });
  const halos: number[][] = new Array(P).fill(null).map((_, p) => {
    const len = rep.partition.halos[p] ?? 0;
    return new Array(len).fill(0);
  });

  return {
    totalMs: rep.pipelineMs,
    graph: {
      nodes,
      edges,
      neighborOffsets: new Uint32Array(N + 1),
      neighborIdx: new Uint32Array(0),
      neighborEdge: new Uint32Array(0),
      buildMs: rep.graph.buildMs,
    },
    features: {
      counts: rep.features.counts,
      symmetryScore: rep.features.symmetryScore,
      minWallThickness: rep.features.minWallThickness,
      avgThinWallThickness: rep.features.avgThinWallThickness,
    },
    manufacturability: {
      feasibility: rep.manufacturability.feasibility,
      machiningAccess: rep.manufacturability.machiningAccess,
      supportFraction: rep.manufacturability.supportFraction,
      thermalDistortionRisk: rep.manufacturability.thermalDistortionRisk,
      assemblyComplexity: rep.manufacturability.assemblyComplexity,
      drivers: rep.manufacturability.drivers,
    },
    priors: {
      timestepScale: rep.priors.timestepScale,
      damping: rep.priors.damping,
      contactStiffness: rep.priors.contactStiffness,
      refinementHints: rep.priors.refinementHints,
    },
    partition: {
      partitionCount: P,
      edgeCut: rep.partition.edgeCut,
      imbalance: rep.partition.imbalance,
      owners: new Int32Array(N),
      resident,
      halos,
      commMatrix: rep.partition.commMatrix.slice(),
    },
    embedding: {
      vector: new Float32Array(rep.embedding.vector),
      slices: rep.embedding.slices,
    },
  } as unknown as TopologyResult;
}
