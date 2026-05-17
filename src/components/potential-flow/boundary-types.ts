/** Shared types for the potential-flow boundary editor + solver panel. */

export type FaceKey = "+x" | "-x" | "+y" | "-y" | "+z" | "-z";
export type FaceMode = "dirichlet" | "neumann" | "wall";

export interface FaceBC {
  mode: FaceMode;
  /** φ value (m²/s) when mode = "dirichlet". */
  phi: number;
  /** Normal velocity v·n_out (m/s) when mode = "neumann". Positive = outflow. */
  vN: number;
}

export const FACE_KEYS: FaceKey[] = ["-x", "+x", "-y", "+y", "-z", "+z"];

export const MODE_LABEL: Record<FaceMode, string> = {
  dirichlet: "Dirichlet φ",
  neumann:   "Neumann v·n",
  wall:      "Wall (no-penetration)",
};
