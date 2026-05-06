// Post-Fabrication Scan & Quality Report Import
// ─────────────────────────────────────────────────────────────
// Parses uploaded CSV / JSON / JSONL inspection reports and maps
// arbitrary column names onto the four fabrication-feedback
// channels (dimensional, thermal, tolerance, surface). Provides a
// tiny pub/sub bridge so the FabFeedbackPanel can auto-ingest any
// imported observations.

import type { Channel, Observation } from "./fabFeedback";

export type RawRow = Record<string, string | number>;

export type FieldMapping = {
  channel: Channel;
  predicted: string;       // raw column name → predicted value
  measured:  string;       // raw column name → measured value
  partId?:   string;       // optional column for part id
  ts?:       string;       // optional column for timestamp
};

// ─── format detection / parsing ─────────────────────────────
export function parseScanFile(name: string, text: string): RawRow[] {
  const lower = name.toLowerCase();
  if (lower.endsWith(".json")) {
    const j = JSON.parse(text);
    if (Array.isArray(j)) return j as RawRow[];
    if (j && Array.isArray(j.rows)) return j.rows as RawRow[];
    if (j && Array.isArray(j.observations)) return j.observations as RawRow[];
    throw new Error("JSON must be an array, or an object with `rows` / `observations`.");
  }
  if (lower.endsWith(".jsonl") || lower.endsWith(".ndjson")) {
    return text
      .split(/\r?\n/)
      .filter((l) => l.trim())
      .map((l) => JSON.parse(l) as RawRow);
  }
  // default: CSV (also handles TSV when separator is auto-detected)
  return parseCSV(text);
}

function parseCSV(text: string): RawRow[] {
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length === 0) return [];
  const sep = lines[0].includes("\t") && !lines[0].includes(",") ? "\t" : ",";
  const splitRow = (l: string) => {
    // simple CSV split with quoted-field support
    const out: string[] = [];
    let cur = "", inQ = false;
    for (let i = 0; i < l.length; i++) {
      const c = l[i];
      if (c === '"') { inQ = !inQ; continue; }
      if (c === sep && !inQ) { out.push(cur); cur = ""; continue; }
      cur += c;
    }
    out.push(cur);
    return out.map((s) => s.trim());
  };
  const headers = splitRow(lines[0]);
  return lines.slice(1).map((line) => {
    const cells = splitRow(line);
    const row: RawRow = {};
    headers.forEach((h, i) => {
      const v = cells[i] ?? "";
      const n = Number(v);
      row[h] = v !== "" && !Number.isNaN(n) ? n : v;
    });
    return row;
  });
}

// ─── fuzzy field auto-detection ─────────────────────────────
// Hints map → which channel is most likely + which role per token.
const CHANNEL_HINTS: Record<Channel, string[]> = {
  dimensional: ["dim", "length", "width", "diameter", "size", "geom", "µm", "um", "mm", "deviation"],
  thermal:     ["temp", "thermal", "heat", "°c", "celsius", "kelvin"],
  tolerance:   ["tol", "spec", "%spec", "ppm", "process_capability", "cpk"],
  surface:     ["surface", "ra", "roughness", "defect", "qa", "porosity", "finish"],
};

const PRED_HINTS = ["predicted", "pred", "expected", "target", "nominal", "design"];
const MEAS_HINTS = ["measured", "meas", "actual", "observed", "scan", "report"];
const PART_HINTS = ["part", "serial", "sn", "id", "uuid", "lot"];
const TS_HINTS   = ["ts", "time", "timestamp", "date", "captured"];

function score(name: string, hints: string[]): number {
  const n = name.toLowerCase();
  let s = 0;
  for (const h of hints) if (n.includes(h)) s += 1;
  return s;
}

export function autoDetectChannel(columns: string[]): Channel {
  let best: Channel = "dimensional";
  let bestScore = -1;
  for (const ch of Object.keys(CHANNEL_HINTS) as Channel[]) {
    const s = columns.reduce((acc, c) => acc + score(c, CHANNEL_HINTS[ch]), 0);
    if (s > bestScore) { bestScore = s; best = ch; }
  }
  return best;
}

export function autoDetectMapping(columns: string[]): FieldMapping {
  const channel = autoDetectChannel(columns);
  const pick = (hints: string[]) => {
    let bestCol = columns[0] ?? "";
    let bestS = -1;
    for (const c of columns) {
      const s = score(c, hints);
      if (s > bestS) { bestS = s; bestCol = c; }
    }
    return bestS > 0 ? bestCol : "";
  };
  return {
    channel,
    predicted: pick(PRED_HINTS) || columns[0] || "",
    measured:  pick(MEAS_HINTS) || columns[1] || columns[0] || "",
    partId:    pick(PART_HINTS) || undefined,
    ts:        pick(TS_HINTS)   || undefined,
  };
}

// ─── apply mapping → Observation[] ──────────────────────────
let _seq = 0;
const uid = () => `scan_${Date.now().toString(36)}_${(_seq++).toString(36)}`;

export type MappingResult = {
  observations: Observation[];
  skipped: number;
  reasons: string[];
};

export function applyMapping(rows: RawRow[], m: FieldMapping): MappingResult {
  const obs: Observation[] = [];
  const reasons: string[] = [];
  let skipped = 0;
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    const p = Number(r[m.predicted]);
    const v = Number(r[m.measured]);
    if (!Number.isFinite(p) || !Number.isFinite(v)) {
      skipped += 1;
      if (reasons.length < 4) reasons.push(`row ${i + 1}: non-numeric predicted/measured`);
      continue;
    }
    const partId = m.partId ? String(r[m.partId] ?? "") : `row-${i + 1}`;
    const tsRaw = m.ts ? r[m.ts] : undefined;
    let ts = Date.now();
    if (tsRaw !== undefined) {
      const t = typeof tsRaw === "number" ? tsRaw : Date.parse(String(tsRaw));
      if (Number.isFinite(t)) ts = t;
    }
    obs.push({
      id: uid(),
      channel: m.channel,
      predicted: p,
      measured: v,
      partId: partId || `row-${i + 1}`,
      ts,
    });
  }
  return { observations: obs, skipped, reasons };
}

// ─── pub/sub bridge to FabFeedbackPanel ─────────────────────
type Listener = (batch: Observation[]) => void;
const listeners = new Set<Listener>();

export const scanImportBridge = {
  publish(batch: Observation[]) {
    for (const l of listeners) l(batch);
  },
  subscribe(l: Listener): () => void {
    listeners.add(l);
    return () => listeners.delete(l);
  },
};
