/**
 * trainingDataStore
 *
 * Browser-side store for user-supplied training corpora. Four channels:
 *   - geometry   (CAD/feature embeddings, JSON or CSV)
 *   - simulation (per-step solver outputs)
 *   - fabrication (process telemetry)
 *   - qa         (inspection / pass-fail results)
 *
 * Each channel keeps a list of "datasets" (one per uploaded file) plus a
 * derived schema (column names, sample counts). Subscribers are notified on
 * change so panels can swap synthetic generators for real data without
 * tight coupling.
 */

export type Channel = "geometry" | "simulation" | "fabrication" | "qa";

export interface DatasetMeta {
  id: string;
  name: string;
  channel: Channel;
  rows: number;
  columns: string[];
  uploadedAt: number;
  bytes: number;
}

export interface Dataset extends DatasetMeta {
  /** Row-oriented numeric/string records. Strings only retained for labels. */
  records: Record<string, number | string>[];
}

type Listener = () => void;

const channels: Record<Channel, Dataset[]> = {
  geometry: [], simulation: [], fabrication: [], qa: [],
};
const listeners = new Set<Listener>();

function emit() { for (const l of listeners) try { l(); } catch { /* ignore */ } }

export const trainingStore = {
  list(channel: Channel): Dataset[] { return channels[channel].slice(); },
  all(): Record<Channel, Dataset[]> {
    return { ...channels };
  },
  add(d: Dataset): void {
    channels[d.channel].push(d);
    emit();
  },
  remove(channel: Channel, id: string): void {
    channels[channel] = channels[channel].filter((d) => d.id !== id);
    emit();
  },
  clear(channel?: Channel): void {
    if (channel) channels[channel] = [];
    else (Object.keys(channels) as Channel[]).forEach((c) => (channels[c] = []));
    emit();
  },
  subscribe(l: Listener): () => void {
    listeners.add(l);
    return () => listeners.delete(l);
  },
  /** Total rows for a channel — useful for "real vs synthetic" badges. */
  rowCount(channel: Channel): number {
    return channels[channel].reduce((s, d) => s + d.rows, 0);
  },
};

// ── Parsers ──────────────────────────────────────────────────────────────

function parseCsv(text: string): { columns: string[]; records: Record<string, number | string>[] } {
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length < 2) throw new Error("CSV needs header + at least one row");
  const cols = lines[0].split(",").map((c) => c.trim());
  const records: Record<string, number | string>[] = [];
  for (let i = 1; i < lines.length; i++) {
    const cells = lines[i].split(",");
    const row: Record<string, number | string> = {};
    for (let j = 0; j < cols.length; j++) {
      const raw = (cells[j] ?? "").trim();
      const n = Number(raw);
      row[cols[j]] = raw !== "" && Number.isFinite(n) ? n : raw;
    }
    records.push(row);
  }
  return { columns: cols, records };
}

function parseJson(text: string): { columns: string[]; records: Record<string, number | string>[] } {
  const data = JSON.parse(text);
  const arr: unknown[] = Array.isArray(data)
    ? data
    : Array.isArray((data as { records?: unknown[] })?.records)
      ? (data as { records: unknown[] }).records
      : Array.isArray((data as { samples?: unknown[] })?.samples)
        ? (data as { samples: unknown[] }).samples
        : [];
  if (arr.length === 0) throw new Error("JSON must be an array, or { records } / { samples }");
  const cols = new Set<string>();
  const records: Record<string, number | string>[] = [];
  for (const r of arr) {
    if (!r || typeof r !== "object") continue;
    const row: Record<string, number | string> = {};
    for (const [k, v] of Object.entries(r as Record<string, unknown>)) {
      cols.add(k);
      if (typeof v === "number" && Number.isFinite(v)) row[k] = v;
      else if (typeof v === "string") row[k] = v;
      else if (typeof v === "boolean") row[k] = v ? 1 : 0;
      else if (Array.isArray(v)) row[k] = JSON.stringify(v);
    }
    records.push(row);
  }
  return { columns: [...cols], records };
}

export async function ingestFile(channel: Channel, file: File): Promise<Dataset> {
  const text = await file.text();
  const lower = file.name.toLowerCase();
  const parsed = lower.endsWith(".json") || lower.endsWith(".jsonl")
    ? parseJson(lower.endsWith(".jsonl")
        ? "[" + text.split(/\r?\n/).filter((l) => l.trim()).join(",") + "]"
        : text)
    : parseCsv(text);
  if (parsed.records.length === 0) throw new Error("No rows parsed");
  const ds: Dataset = {
    id: `${channel}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    name: file.name,
    channel,
    rows: parsed.records.length,
    columns: parsed.columns,
    bytes: file.size,
    uploadedAt: Date.now(),
    records: parsed.records,
  };
  trainingStore.add(ds);
  return ds;
}
