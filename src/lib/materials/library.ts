/**
 * Built-in material library.
 *
 * A small, curated set covering common metals, polymers, composites,
 * ceramics, and elastomers. Values are typical engineering handbook
 * numbers — fine for ranking / metric-space reasoning, not for final
 * certification. The engine can also pull rows from the `materials`
 * table; this library is the fallback when the DB is empty.
 */

import type { MaterialRecord } from "./types";

export const BUILTIN_MATERIALS: MaterialRecord[] = [
  // -------- Steels --------
  {
    id: "steel-1018", name: "Steel 1018 (low carbon)", family: "metal",
    density: 7.87, youngsModulus: 200, yieldStrength: 370, ultimateStrength: 440,
    fatigueLimit: 220, thermalConductivity: 52, thermalExpansion: 11.7,
    maxServiceTempC: 425, costPerKg: 1.0, embodiedCO2: 1.9,
    corrosionResistance: 0.25, weatherResistance: 0.3,
    fabrication: ["machining", "casting", "sheet_forming", "extrusion"],
  },
  {
    id: "steel-4340", name: "Steel 4340 (high-strength)", family: "metal",
    density: 7.85, youngsModulus: 205, yieldStrength: 910, ultimateStrength: 1110,
    fatigueLimit: 540, thermalConductivity: 44, thermalExpansion: 12.3,
    maxServiceTempC: 500, costPerKg: 1.8, embodiedCO2: 2.2,
    corrosionResistance: 0.25, weatherResistance: 0.3,
    fabrication: ["machining", "casting"],
  },
  {
    id: "ss-316l", name: "Stainless 316L", family: "metal",
    density: 8.0, youngsModulus: 193, yieldStrength: 290, ultimateStrength: 580,
    fatigueLimit: 240, thermalConductivity: 16, thermalExpansion: 16.0,
    maxServiceTempC: 870, costPerKg: 5.5, embodiedCO2: 5.4,
    corrosionResistance: 0.95, weatherResistance: 0.95,
    fabrication: ["machining", "casting", "additive_dmls", "sheet_forming"],
  },
  // -------- Aluminums --------
  {
    id: "al-6061-t6", name: "Aluminum 6061-T6", family: "metal",
    density: 2.70, youngsModulus: 68.9, yieldStrength: 276, ultimateStrength: 310,
    fatigueLimit: 96, thermalConductivity: 167, thermalExpansion: 23.6,
    maxServiceTempC: 200, costPerKg: 3.5, embodiedCO2: 8.2,
    corrosionResistance: 0.7, weatherResistance: 0.75,
    fabrication: ["machining", "casting", "extrusion", "sheet_forming"],
  },
  {
    id: "al-7075-t6", name: "Aluminum 7075-T6", family: "metal",
    density: 2.81, youngsModulus: 71.7, yieldStrength: 503, ultimateStrength: 572,
    fatigueLimit: 159, thermalConductivity: 130, thermalExpansion: 23.4,
    maxServiceTempC: 175, costPerKg: 6.0, embodiedCO2: 9.0,
    corrosionResistance: 0.55, weatherResistance: 0.6,
    fabrication: ["machining", "extrusion"],
  },
  // -------- Titanium --------
  {
    id: "ti-6al4v", name: "Titanium Ti-6Al-4V", family: "metal",
    density: 4.43, youngsModulus: 113.8, yieldStrength: 880, ultimateStrength: 950,
    fatigueLimit: 510, thermalConductivity: 6.7, thermalExpansion: 8.6,
    maxServiceTempC: 400, costPerKg: 35.0, embodiedCO2: 28.0,
    corrosionResistance: 0.98, weatherResistance: 0.95,
    fabrication: ["machining", "additive_dmls"],
  },
  // -------- Polymers --------
  {
    id: "abs", name: "ABS", family: "polymer",
    density: 1.04, youngsModulus: 2.3, yieldStrength: 41, ultimateStrength: 45,
    fatigueLimit: 12, thermalConductivity: 0.17, thermalExpansion: 90,
    maxServiceTempC: 85, costPerKg: 2.5, embodiedCO2: 3.1,
    corrosionResistance: 0.85, weatherResistance: 0.4,
    fabrication: ["injection_molding", "additive_fdm", "machining"],
  },
  {
    id: "peek", name: "PEEK", family: "polymer",
    density: 1.32, youngsModulus: 3.6, yieldStrength: 100, ultimateStrength: 110,
    fatigueLimit: 28, thermalConductivity: 0.25, thermalExpansion: 47,
    maxServiceTempC: 250, costPerKg: 95.0, embodiedCO2: 8.5,
    corrosionResistance: 0.95, weatherResistance: 0.9,
    fabrication: ["injection_molding", "machining", "additive_fdm"],
  },
  {
    id: "nylon66-gf30", name: "Nylon 6,6 + 30% GF", family: "composite",
    density: 1.36, youngsModulus: 9.0, yieldStrength: 160, ultimateStrength: 180,
    fatigueLimit: 50, thermalConductivity: 0.3, thermalExpansion: 35,
    maxServiceTempC: 150, costPerKg: 6.0, embodiedCO2: 7.0,
    corrosionResistance: 0.85, weatherResistance: 0.7,
    fabrication: ["injection_molding", "machining"],
  },
  // -------- Composites --------
  {
    id: "cfrp-uni", name: "CFRP (unidirectional)", family: "composite",
    density: 1.55, youngsModulus: 135, yieldStrength: 1500, ultimateStrength: 1500,
    fatigueLimit: 800, thermalConductivity: 5, thermalExpansion: 0.5,
    maxServiceTempC: 150, costPerKg: 60.0, embodiedCO2: 24.0,
    corrosionResistance: 0.9, weatherResistance: 0.8,
    fabrication: ["layup", "additive_fdm"],
  },
  {
    id: "gfrp", name: "GFRP (E-glass / epoxy)", family: "composite",
    density: 2.0, youngsModulus: 35, yieldStrength: 500, ultimateStrength: 700,
    fatigueLimit: 250, thermalConductivity: 0.3, thermalExpansion: 8,
    maxServiceTempC: 130, costPerKg: 8.0, embodiedCO2: 5.0,
    corrosionResistance: 0.9, weatherResistance: 0.85,
    fabrication: ["layup"],
  },
  // -------- Ceramics --------
  {
    id: "alumina-995", name: "Alumina 99.5%", family: "ceramic",
    density: 3.96, youngsModulus: 370, yieldStrength: 300, ultimateStrength: 300,
    fatigueLimit: 200, thermalConductivity: 30, thermalExpansion: 8.1,
    maxServiceTempC: 1750, costPerKg: 12.0, embodiedCO2: 5.5,
    corrosionResistance: 0.98, weatherResistance: 0.95,
    fabrication: ["sintering", "machining"],
  },
  {
    id: "sic", name: "Silicon Carbide (SiC)", family: "ceramic",
    density: 3.21, youngsModulus: 410, yieldStrength: 350, ultimateStrength: 550,
    fatigueLimit: 320, thermalConductivity: 120, thermalExpansion: 4.0,
    maxServiceTempC: 1600, costPerKg: 50.0, embodiedCO2: 9.0,
    corrosionResistance: 0.97, weatherResistance: 0.95,
    fabrication: ["sintering", "machining"],
  },
  // -------- Elastomer --------
  {
    id: "epdm", name: "EPDM rubber", family: "elastomer",
    density: 1.10, youngsModulus: 0.01, yieldStrength: 14, ultimateStrength: 17,
    fatigueLimit: 5, thermalConductivity: 0.35, thermalExpansion: 200,
    maxServiceTempC: 150, costPerKg: 4.0, embodiedCO2: 3.2,
    corrosionResistance: 0.85, weatherResistance: 0.95,
    fabrication: ["injection_molding", "extrusion"],
  },
];
