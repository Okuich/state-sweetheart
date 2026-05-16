/**
 * Sparse linear algebra (CSR) for the Laplacian Physics Engine.
 *
 * Pure TypeScript, deterministic, dependency-free. All routines operate on
 * Float64Array vectors and CSR matrices so they remain GPU-portable
 * (matching layouts the WebGPU backend can consume directly).
 */

export interface CSRMatrix {
  /** Square matrix dimension. */
  n: number;
  /** Row pointer array of length n+1. */
  rowPtr: Int32Array;
  /** Column index per non-zero, length = nnz. */
  colIdx: Int32Array;
  /** Value per non-zero, length = nnz. */
  values: Float64Array;
}

/**
 * Build a CSR matrix from a coordinate-list (COO) of (row, col, value).
 * Duplicates on the same (row, col) are summed — needed for FEM assembly.
 */
export function buildCSR(
  n: number,
  triplets: ReadonlyArray<readonly [number, number, number]>,
): CSRMatrix {
  // Bucket sort by row, then merge duplicate (row,col).
  const counts = new Int32Array(n);
  for (const [r] of triplets) counts[r]++;
  const rowPtr = new Int32Array(n + 1);
  for (let i = 0; i < n; i++) rowPtr[i + 1] = rowPtr[i] + counts[i];
  const cursor = new Int32Array(rowPtr.subarray(0, n));
  const tmpCol = new Int32Array(triplets.length);
  const tmpVal = new Float64Array(triplets.length);
  for (const [r, c, v] of triplets) {
    const k = cursor[r]++;
    tmpCol[k] = c;
    tmpVal[k] = v;
  }
  // Sort within each row, merging duplicates.
  const outCol: number[] = [];
  const outVal: number[] = [];
  const newRowPtr = new Int32Array(n + 1);
  for (let r = 0; r < n; r++) {
    const start = rowPtr[r];
    const end = rowPtr[r + 1];
    const idx: number[] = [];
    for (let k = start; k < end; k++) idx.push(k);
    idx.sort((a, b) => tmpCol[a] - tmpCol[b]);
    let lastCol = -1;
    for (const k of idx) {
      const c = tmpCol[k];
      const v = tmpVal[k];
      if (c === lastCol) {
        outVal[outVal.length - 1] += v;
      } else {
        outCol.push(c);
        outVal.push(v);
        lastCol = c;
      }
    }
    newRowPtr[r + 1] = outCol.length;
  }
  return {
    n,
    rowPtr: newRowPtr,
    colIdx: Int32Array.from(outCol),
    values: Float64Array.from(outVal),
  };
}

/** y = A · x. */
export function spmv(A: CSRMatrix, x: Float64Array, y: Float64Array): void {
  const { n, rowPtr, colIdx, values } = A;
  for (let r = 0; r < n; r++) {
    let s = 0;
    const end = rowPtr[r + 1];
    for (let k = rowPtr[r]; k < end; k++) s += values[k] * x[colIdx[k]];
    y[r] = s;
  }
}

/** Return the diagonal of A as a dense vector. */
export function diag(A: CSRMatrix): Float64Array {
  const d = new Float64Array(A.n);
  const { rowPtr, colIdx, values } = A;
  for (let r = 0; r < A.n; r++) {
    for (let k = rowPtr[r]; k < rowPtr[r + 1]; k++) {
      if (colIdx[k] === r) {
        d[r] = values[k];
        break;
      }
    }
  }
  return d;
}

/** dot(a, b). */
export function dot(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** y += α · x. */
export function axpy(alpha: number, x: Float64Array, y: Float64Array): void {
  for (let i = 0; i < x.length; i++) y[i] += alpha * x[i];
}

/** y = α · x + β · y. */
export function axpby(
  alpha: number,
  x: Float64Array,
  beta: number,
  y: Float64Array,
): void {
  for (let i = 0; i < x.length; i++) y[i] = alpha * x[i] + beta * y[i];
}

/** Euclidean norm. */
export function norm2(a: Float64Array): number {
  return Math.sqrt(dot(a, a));
}

/** Pin row/col `i` to identity with rhs = `value`. In-place. */
export function applyDirichlet(
  A: CSRMatrix,
  rhs: Float64Array,
  i: number,
  value: number,
): void {
  const { rowPtr, colIdx, values } = A;
  // Zero off-diagonals in row i, set diagonal to 1.
  for (let k = rowPtr[i]; k < rowPtr[i + 1]; k++) {
    values[k] = colIdx[k] === i ? 1 : 0;
  }
  // Eliminate column i from other rows (move to rhs).
  for (let r = 0; r < A.n; r++) {
    if (r === i) continue;
    for (let k = rowPtr[r]; k < rowPtr[r + 1]; k++) {
      if (colIdx[k] === i) {
        rhs[r] -= values[k] * value;
        values[k] = 0;
        break;
      }
    }
  }
  rhs[i] = value;
}
