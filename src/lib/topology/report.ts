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

const FEATURE_ORDER: FeatureClass[] = [
  "bulk", "boundary", "thin_wall", "overhang",
  "cavity", "stress_concentrator", "thermal_bottleneck", "symmetry_seed",
];

export interface TopologyReport {
  generatedAt: string;
  pipelineMs: number;
  graph: {
    nodes: number;
    edges: number;
    meanValence: number;
    buildMs: number;
  };
  features: {
    counts: Record<FeatureClass, number>;
    symmetryScore: number;
    minWallThickness: number;
    avgThinWallThickness: number;
  };
  manufacturability: {
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
  priors: {
    timestepScale: number;
    damping: number;
    contactStiffness: number;
    refinementHints: Record<FeatureClass, number>;
  };
  partition: {
    partitionCount: number;
    edgeCut: number;
    imbalance: number;
    resident: number[];
    halos: number[];
    commMatrix: number[];
  };
  embedding: {
    dim: number;
    slices: { curvature: [number, number]; features: [number, number]; structural: [number, number]; manuf: [number, number] };
    vector: number[];
  };
}

export function buildReportJSON(r: TopologyResult): TopologyReport {
  const meanValence = (r.graph.edges.length * 2) / Math.max(1, r.graph.nodes.length);
  return {
    generatedAt: new Date().toISOString(),
    pipelineMs: r.totalMs,
    graph: {
      nodes: r.graph.nodes.length,
      edges: r.graph.edges.length,
      meanValence,
      buildMs: r.graph.buildMs,
    },
    features: {
      counts: r.features.counts,
      symmetryScore: r.features.symmetryScore,
      minWallThickness: r.features.minWallThickness,
      avgThinWallThickness: r.features.avgThinWallThickness,
    },
    manufacturability: {
      feasibility: r.manufacturability.feasibility,
      machiningAccess: r.manufacturability.machiningAccess,
      supportFraction: r.manufacturability.supportFraction,
      thermalDistortionRisk: r.manufacturability.thermalDistortionRisk,
      assemblyComplexity: r.manufacturability.assemblyComplexity,
      drivers: r.manufacturability.drivers,
    },
    priors: {
      timestepScale: r.priors.timestepScale,
      damping: r.priors.damping,
      contactStiffness: r.priors.contactStiffness,
      refinementHints: r.priors.refinementHints,
    },
    partition: {
      partitionCount: r.partition.partitionCount,
      edgeCut: r.partition.edgeCut,
      imbalance: r.partition.imbalance,
      resident: r.partition.resident.map((p) => p.length),
      halos: r.partition.halos.map((p) => p.length),
      commMatrix: Array.from(r.partition.commMatrix),
    },
    embedding: {
      dim: r.embedding.vector.length,
      slices: r.embedding.slices,
      vector: Array.from(r.embedding.vector),
    },
  };
}

export function buildReportPDF(r: TopologyResult, label?: string): Blob {
  const rep = buildReportJSON(r);
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
    doc.setDrawColor(220); doc.setFillColor(235);
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
  for (const c of FEATURE_ORDER) bar(FEATURE_LABELS[c], rep.features.counts[c], Math.max(1, rep.graph.nodes));

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
  const cellW = Math.min(36, (W - M * 2 - 30) / P);
  const cellH = 14;
  const maxFlow = Math.max(1, ...rep.partition.commMatrix);
  ensure(cellH * (P + 1) + 10);
  doc.setFontSize(8);
  for (let c = 0; c < P; c++) doc.text(`P${c}`, M + 30 + c * cellW + cellW / 2 - 4, y);
  y += 10;
  for (let r2 = 0; r2 < P; r2++) {
    doc.text(`P${r2}`, M, y + cellH - 4);
    for (let c = 0; c < P; c++) {
      const v = rep.partition.commMatrix[r2 * P + c];
      const t = v / maxFlow;
      doc.setFillColor(255 - Math.round(t * 195), 255 - Math.round(t * 145), 255 - Math.round(t * 55));
      doc.rect(M + 30 + c * cellW, y, cellW - 2, cellH - 2, "F");
      if (v > 0) {
        doc.setTextColor(t > 0.55 ? 255 : 30);
        doc.text(`${v}`, M + 30 + c * cellW + cellW / 2 - 4, y + cellH - 5);
      }
    }
    y += cellH;
  }
  doc.setTextColor(20);
  y += 6;

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
