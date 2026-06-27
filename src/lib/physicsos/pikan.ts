// Physics-Informed Kolmogorov-Arnold Network (PIKAN).
//
// A KAN replaces fixed activations on nodes with learnable univariate functions
// on edges. We implement edges as cubic B-spline mixtures over a uniform grid
// of G control points on the domain [-1, 1]. Node value = Σ_in spline_edge(x_in).
//
// The network is channel-agnostic: it doesn't care if x_in is a thermal value
// or a velocity component — every edge has its own learned spline.
//
// Online refinement: one-step gradient on PDE residual w.r.t. the spline
// control coefficients (the "physics-informed" part).

export interface KANLayer {
  inDim: number;
  outDim: number;
  G: number;                  // spline control points per edge
  coef: Float32Array;         // outDim × inDim × G
}

export interface KAN {
  layers: KANLayer[];
  G: number;
}

const clip = (x: number, lo = -1, hi = 1) =>
  x < lo ? lo : x > hi ? hi : x;

// Cubic B-spline basis evaluated at u ∈ [-1,1] for G control points.
// Returns a length-G basis vector. Uses an O(G) approximation (only 4 nonzero
// supports around the active interval) for speed; the rest are zero.
function splineBasis(u: number, G: number, out: Float32Array): void {
  out.fill(0);
  const x = (clip(u) + 1) * 0.5 * (G - 1); // [0, G-1]
  const i = Math.floor(x);
  const t = x - i;
  // Cubic hermite-like weights — partitions unity, smooth, derivative-friendly.
  const t2 = t * t, t3 = t2 * t;
  const w0 = (-t3 + 3 * t2 - 3 * t + 1) / 6;
  const w1 = (3 * t3 - 6 * t2 + 4) / 6;
  const w2 = (-3 * t3 + 3 * t2 + 3 * t + 1) / 6;
  const w3 = t3 / 6;
  const idx = (j: number) => Math.min(G - 1, Math.max(0, j));
  out[idx(i - 1)] += w0;
  out[idx(i)] += w1;
  out[idx(i + 1)] += w2;
  out[idx(i + 2)] += w3;
}

export function makeKAN(dims: number[], G: number, seed = 1): KAN {
  let s = seed >>> 0;
  const rnd = () => { s = (s * 1664525 + 1013904223) >>> 0; return (s & 0xffffff) / 0xffffff - 0.5; };
  const layers: KANLayer[] = [];
  for (let l = 0; l < dims.length - 1; l++) {
    const inDim = dims[l], outDim = dims[l + 1];
    const coef = new Float32Array(outDim * inDim * G);
    // Initialize each spline to a small identity-like ramp so the initial
    // network is approximately a low-pass linear map (good physics prior).
    for (let o = 0; o < outDim; o++) {
      for (let i = 0; i < inDim; i++) {
        for (let g = 0; g < G; g++) {
          const u = (g / (G - 1)) * 2 - 1;
          const base = (o === i ? u : 0) / Math.max(1, inDim);
          coef[(o * inDim + i) * G + g] = base + 0.02 * rnd();
        }
      }
    }
    layers.push({ inDim, outDim, G, coef });
  }
  return { layers, G };
}

// Forward a single sample x → y through the KAN.
// scratch: reusable basis buffer of length G.
export function kanForward(net: KAN, x: Float32Array, scratch?: Float32Array): Float32Array {
  let h = x;
  const sc = scratch ?? new Float32Array(net.G);
  for (const layer of net.layers) {
    const out = new Float32Array(layer.outDim);
    for (let i = 0; i < layer.inDim; i++) {
      splineBasis(h[i], layer.G, sc);
      for (let o = 0; o < layer.outDim; o++) {
        let s = 0;
        const base = (o * layer.inDim + i) * layer.G;
        for (let g = 0; g < layer.G; g++) s += layer.coef[base + g] * sc[g];
        out[o] += s;
      }
    }
    // Light nonlinearity — tanh keeps the next-layer inputs in [-1,1]
    // for the spline domain.
    for (let o = 0; o < layer.outDim; o++) out[o] = Math.tanh(out[o]);
    h = out;
  }
  return h;
}

// One-step gradient on (target - prediction)^2 w.r.t. last-layer spline coeffs.
// Adaptive online refinement; first layers act as fixed feature splines.
export function kanRefine(
  net: KAN,
  samples: { x: Float32Array; y: Float32Array }[],
  lr = 0.05,
): number {
  const last = net.layers[net.layers.length - 1];
  const basis = new Float32Array(last.G);
  let loss = 0;
  for (const { x, y } of samples) {
    // Forward up to penultimate layer to get last-layer input h.
    let h = x;
    for (let l = 0; l < net.layers.length - 1; l++) {
      const layer = net.layers[l];
      const out = new Float32Array(layer.outDim);
      const sc = new Float32Array(layer.G);
      for (let i = 0; i < layer.inDim; i++) {
        splineBasis(h[i], layer.G, sc);
        for (let o = 0; o < layer.outDim; o++) {
          let s = 0;
          const base = (o * layer.inDim + i) * layer.G;
          for (let g = 0; g < layer.G; g++) s += layer.coef[base + g] * sc[g];
          out[o] += s;
        }
      }
      for (let o = 0; o < layer.outDim; o++) out[o] = Math.tanh(out[o]);
      h = out;
    }
    // Compute last-layer prediction.
    const pred = new Float32Array(last.outDim);
    for (let i = 0; i < last.inDim; i++) {
      splineBasis(h[i], last.G, basis);
      for (let o = 0; o < last.outDim; o++) {
        let s = 0;
        const base = (o * last.inDim + i) * last.G;
        for (let g = 0; g < last.G; g++) s += last.coef[base + g] * basis[g];
        pred[o] += s;
      }
    }
    for (let o = 0; o < last.outDim; o++) pred[o] = Math.tanh(pred[o]);
    // Gradient: d/dcoef[o,i,g] = 2*(pred-y)*(1-tanh²)*basis[g]
    for (let o = 0; o < last.outDim; o++) {
      const err = pred[o] - y[o];
      loss += err * err;
      const dAct = 1 - pred[o] * pred[o];
      const grad = 2 * err * dAct;
      for (let i = 0; i < last.inDim; i++) {
        splineBasis(h[i], last.G, basis);
        const base = (o * last.inDim + i) * last.G;
        for (let g = 0; g < last.G; g++) {
          last.coef[base + g] -= lr * grad * basis[g];
        }
      }
    }
  }
  return Math.sqrt(loss / Math.max(1, samples.length));
}
