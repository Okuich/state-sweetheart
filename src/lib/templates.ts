// Domain Simulation Template System
// ─────────────────────────────────────────────────────────────────────
// High-level templates that compile down to concrete SimParams patches.
// Each template declares its physical domain, a small set of meaningful
// knobs (e.g. "wall temperature", "load magnitude"), and a `compile()`
// function that turns those knobs + a chosen material into integrator,
// stiffness, damping, field, etc. settings.
//
// A material library is shared across templates so swapping "Al-7075"
// for "CFRP" automatically rescales spring stiffness, mass, and damping
// in any template that consumes it.

import type { SimParams } from "@/components/PhysicsCanvas";

export type Domain = "fabrication" | "manufacturing" | "thermal" | "stress" | "flow";

// ── Material library ────────────────────────────────────────────────────
// Properties are normalized for the visual sandbox — they drive *relative*
// stiffness, density and damping rather than SI values. A reference
// material (Al-7075) is tuned to look right at the default scene scale;
// other materials scale around it.
export type Material = {
  id: string;
  label: string;
  density: number;       // kg/m³ — used as relative mass multiplier
  young: number;         // GPa  — drives spring stiffness
  damping: number;       // intrinsic loss factor (0..1)
  thermalK: number;      // W/m·K — only used by thermal templates
  yieldMPa: number;      // only used by stress templates
  color: string;         // visual hint
};

export const MATERIALS: Material[] = [
  { id: "al7075",   label: "Al-7075",       density: 2810, young:  72, damping: 0.02, thermalK: 130, yieldMPa: 503, color: "#cfd6dd" },
  { id: "ti6al4v",  label: "Ti-6Al-4V",     density: 4430, young: 114, damping: 0.01, thermalK:   7, yieldMPa: 880, color: "#d8c79a" },
  { id: "ss316",    label: "Stainless 316", density: 7990, young: 193, damping: 0.03, thermalK:  16, yieldMPa: 290, color: "#aab1bb" },
  { id: "cfrp",     label: "CFRP",          density: 1600, young: 230, damping: 0.08, thermalK:   7, yieldMPa: 600, color: "#1a1a1f" },
  { id: "abs",      label: "ABS plastic",   density: 1050, young:   2, damping: 0.20, thermalK: 0.2, yieldMPa:  40, color: "#e6d6b8" },
  { id: "water",    label: "Water (fluid)", density: 1000, young:   0, damping: 0.50, thermalK: 0.6, yieldMPa:   0, color: "#6ba8d4" },
  { id: "air",      label: "Air (fluid)",   density:    1, young:   0, damping: 0.05, thermalK: 0.026, yieldMPa: 0, color: "#cfe6f7" },
];

export function getMaterial(id: string): Material {
  return MATERIALS.find((m) => m.id === id) ?? MATERIALS[0];
}

// ── Knob schema ─────────────────────────────────────────────────────────
// Each template declares a few high-level knobs the user actually cares
// about. The TemplatePanel renders them as sliders/selects automatically.
export type Knob =
  | { id: string; label: string; kind: "number"; min: number; max: number; step: number; default: number; unit?: string }
  | { id: string; label: string; kind: "select"; options: { value: string; label: string }[]; default: string };

export type KnobValues = Record<string, number | string>;

// ── Template definition ─────────────────────────────────────────────────
export type Template = {
  id: string;
  domain: Domain;
  label: string;
  description: string;
  // Material slot(s). Most templates take one; flow takes a fluid + solid.
  materialSlots: { id: string; label: string; defaults: string[] }[];
  knobs: Knob[];
  // Solver/integrator recommendation. The compiler will copy these into
  // the resulting SimParams patch, but a user can still override.
  solver: {
    integrator: SimParams["integrator"];
    pairwiseAlgo: SimParams["pairwiseAlgo"];
    pairwiseMode: SimParams["pairwiseMode"];
    boundary: SimParams["boundary"];
  };
  compile: (knobs: KnobValues, materials: Record<string, Material>) => Partial<SimParams>;
};

// Helper: read a numeric knob with a fallback so templates stay terse.
function n(k: KnobValues, id: string, fallback: number): number {
  const v = k[id]; return typeof v === "number" ? v : fallback;
}
function s(k: KnobValues, id: string, fallback: string): string {
  const v = k[id]; return typeof v === "string" ? v : fallback;
}

// ── Templates ───────────────────────────────────────────────────────────
const fabrication: Template = {
  id: "additive_print",
  domain: "fabrication",
  label: "Additive print bed",
  description:
    "Layer-by-layer deposition with thermal damping. Particles model deposited beads cooling onto a substrate.",
  materialSlots: [{ id: "feedstock", label: "Feedstock", defaults: ["abs", "al7075"] }],
  knobs: [
    { id: "layerHeight",  label: "Layer height", kind: "number", min: 1, max: 12, step: 0.5, default: 4, unit: "px" },
    { id: "depositionRate", label: "Deposition rate", kind: "number", min: 50, max: 800, step: 10, default: 300, unit: "p/s" },
    { id: "bedTemp",      label: "Bed temperature", kind: "number", min: 20, max: 200, step: 5, default: 80, unit: "°C" },
  ],
  solver: { integrator: "verlet", pairwiseAlgo: "grid", pairwiseMode: "lj", boundary: "walls" },
  compile: (k, m) => {
    const mat = m.feedstock;
    const tempFactor = 1 + (n(k, "bedTemp", 80) - 20) / 200; // hotter = looser
    return {
      particleCount: Math.round(n(k, "depositionRate", 300)),
      restLength: n(k, "layerHeight", 4) * 6,
      springK: 60 * (mat.young / 72),
      damping: Math.min(0.95, mat.damping * 8 * tempFactor),
      gravity: 80,
      pairwiseStrength: 120,
      pairwiseRadius: 40,
    };
  },
};

const manufacturing: Template = {
  id: "conveyor_packing",
  domain: "manufacturing",
  label: "Conveyor packing line",
  description:
    "Discrete part flow on a moving belt. Wrap boundary models a continuous loop; stronger pairwise repulsion = denser packing.",
  materialSlots: [{ id: "part", label: "Part material", defaults: ["abs", "ss316"] }],
  knobs: [
    { id: "throughput", label: "Throughput", kind: "number", min: 100, max: 1500, step: 50, default: 600, unit: "p/min" },
    { id: "beltSpeed",  label: "Belt speed",  kind: "number", min: 0, max: 200, step: 5, default: 60, unit: "px/s" },
    { id: "packMode",   label: "Pack mode",   kind: "select", default: "loose", options: [
      { value: "loose", label: "Loose" }, { value: "dense", label: "Dense" }, { value: "jammed", label: "Jammed" },
    ] },
  ],
  solver: { integrator: "semi-euler", pairwiseAlgo: "grid", pairwiseMode: "repel", boundary: "wrap" },
  compile: (k, m) => {
    const mode = s(k, "packMode", "loose");
    const repel = mode === "jammed" ? 380 : mode === "dense" ? 220 : 120;
    return {
      particleCount: Math.round(n(k, "throughput", 600)),
      gravity: n(k, "beltSpeed", 60) * 0.5,
      pairwiseStrength: repel,
      pairwiseRadius: mode === "jammed" ? 35 : 55,
      damping: 0.3 + m.part.damping * 2,
      restLength: 30,
      springK: 0,
    };
  },
};

const thermal: Template = {
  id: "thermal_diffusion",
  domain: "thermal",
  label: "Thermal diffusion plate",
  description:
    "Heat propagates as kinetic energy across a soft lattice. Material's thermalK rescales effective diffusivity.",
  materialSlots: [{ id: "plate", label: "Plate material", defaults: ["al7075", "ss316"] }],
  knobs: [
    { id: "wallTemp", label: "Wall ΔT", kind: "number", min: 0, max: 500, step: 10, default: 200, unit: "K" },
    { id: "meshDensity", label: "Mesh density", kind: "number", min: 100, max: 1200, step: 50, default: 500 },
    { id: "ambient", label: "Ambient", kind: "select", default: "vacuum", options: [
      { value: "vacuum", label: "Vacuum" }, { value: "convective", label: "Convective" },
    ] },
  ],
  solver: { integrator: "verlet", pairwiseAlgo: "grid", pairwiseMode: "lj", boundary: "walls" },
  compile: (k, m) => {
    const ambient = s(k, "ambient", "vacuum");
    return {
      particleCount: Math.round(n(k, "meshDensity", 500)),
      gravity: n(k, "wallTemp", 200) * 0.2, // proxy: drives motion magnitude
      damping: ambient === "convective" ? 0.4 : Math.max(0.02, 0.5 / m.plate.thermalK),
      springK: 40 * (m.plate.young / 72),
      restLength: 22,
      edgesPerNode: 4,
      constraintIters: 4,
      field: "ripple",
      fieldStrength: 0.3,
    };
  },
};

const stress: Template = {
  id: "tensile_load",
  domain: "stress",
  label: "Tensile load test",
  description:
    "Spring lattice under uniaxial pull. Load magnitude becomes external gravity along the pull axis; springK derives from Young's modulus.",
  materialSlots: [{ id: "specimen", label: "Specimen", defaults: ["al7075", "ti6al4v", "cfrp"] }],
  knobs: [
    { id: "load", label: "Load", kind: "number", min: 0, max: 400, step: 10, default: 120, unit: "MPa" },
    { id: "nodes", label: "Mesh nodes", kind: "number", min: 100, max: 1000, step: 50, default: 400 },
    { id: "boundary", label: "BC", kind: "select", default: "walls", options: [
      { value: "walls", label: "Clamped" }, { value: "wrap", label: "Periodic" },
    ] },
  ],
  solver: { integrator: "verlet", pairwiseAlgo: "grid", pairwiseMode: "lj", boundary: "walls" },
  compile: (k, m) => {
    const safety = m.specimen.yieldMPa / Math.max(1, n(k, "load", 120));
    return {
      particleCount: Math.round(n(k, "nodes", 400)),
      gravity: n(k, "load", 120) * 0.6,
      springK: 60 * (m.specimen.young / 72),
      restLength: 28,
      edgesPerNode: 3,
      constraintIters: Math.min(10, Math.round(8 / Math.max(0.5, safety))),
      damping: m.specimen.damping * 5,
      boundary: s(k, "boundary", "walls") as SimParams["boundary"],
      pairwiseStrength: 60,
    };
  },
};

const flow: Template = {
  id: "lid_driven_cavity",
  domain: "flow",
  label: "Lid-driven cavity (SPH-lite)",
  description:
    "Particle-based fluid in a closed cavity. Top boundary drags particles; viscosity from the fluid material's damping.",
  materialSlots: [
    { id: "fluid", label: "Fluid", defaults: ["water", "air"] },
    { id: "wall",  label: "Wall material", defaults: ["ss316", "al7075"] },
  ],
  knobs: [
    { id: "lidSpeed", label: "Lid speed", kind: "number", min: 0, max: 300, step: 5, default: 120, unit: "px/s" },
    { id: "particles", label: "SPH particles", kind: "number", min: 200, max: 1500, step: 50, default: 800 },
    { id: "smoothing", label: "Smoothing radius", kind: "number", min: 20, max: 90, step: 5, default: 45, unit: "px" },
  ],
  solver: { integrator: "semi-euler", pairwiseAlgo: "grid", pairwiseMode: "repel", boundary: "walls" },
  compile: (k, m) => {
    const re = (n(k, "lidSpeed", 120) * 100) / Math.max(0.01, m.fluid.damping * 100);
    return {
      particleCount: Math.round(n(k, "particles", 800)),
      pairwiseRadius: n(k, "smoothing", 45),
      pairwiseStrength: 200 + (m.fluid.density / 100),
      gravity: 30,
      damping: Math.min(0.9, m.fluid.damping * 1.5),
      springK: 0,
      restLength: 24,
      field: "swirl",
      fieldStrength: Math.min(2, re / 2000),
    };
  },
};

export const TEMPLATES: Template[] = [fabrication, manufacturing, thermal, stress, flow];

export function getTemplate(id: string): Template {
  return TEMPLATES.find((t) => t.id === id) ?? TEMPLATES[0];
}

// Build the full SimParams patch for a template instance: knob values +
// chosen material per slot + the template's solver recommendation.
export function compileTemplate(
  tmpl: Template,
  knobs: KnobValues,
  materialIds: Record<string, string>,
): Partial<SimParams> {
  const materials: Record<string, Material> = {};
  for (const slot of tmpl.materialSlots) {
    materials[slot.id] = getMaterial(materialIds[slot.id] ?? slot.defaults[0]);
  }
  const physics = tmpl.compile(knobs, materials);
  return { ...physics, ...tmpl.solver };
}

export function defaultKnobs(tmpl: Template): KnobValues {
  const out: KnobValues = {};
  for (const k of tmpl.knobs) out[k.id] = k.default;
  return out;
}
export function defaultMaterials(tmpl: Template): Record<string, string> {
  const out: Record<string, string> = {};
  for (const slot of tmpl.materialSlots) out[slot.id] = slot.defaults[0];
  return out;
}
