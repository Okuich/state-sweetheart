## Unified Laplacian Physics Engine — Phased Plan

A massive 9-phase build. Proposing one chat turn per phase so you can verify each layer before the next, matching the existing project rhythm (see `.lovable/plan.md`).

### Phase 1 — PDE Core (next turn)
New module `src/lib/pde/`:
- `sparse.ts` — CSR sparse matrix, SpMV, axpy, dot
- `laplacian.ts` — FEM (P1 tet/hex) + FVM Laplace operator assembly over Geometry OS `OctreeMesh`
- `poisson.ts` — Poisson RHS assembly, Dirichlet/Neumann BCs
- `solvers/cg.ts` — preconditioned conjugate gradient
- `solvers/multigrid.ts` — geometric multigrid V-cycle on octree levels
- `solvers/jacobi.ts` — smoother
- `index.ts` + tests (`pde.test.ts`)
- Hooks into `src/lib/sdf` and `src/lib/meshing` for operator stencils.

### Phase 2 — Thermal field engine
`src/lib/pde/thermal.ts` + `ThermalFieldPanel.tsx`. Steady-state heat, anisotropic κ tensor, flux BCs. Outputs T(x), hotspot map, thermal-stress proxy. Wired as a server fn `solveThermal`.

### Phase 3 — Electrostatic engine
`src/lib/pde/electrostatic.ts` + `ElectroFieldPanel.tsx`. Permittivity, ρ, V boundaries. Outputs V, E = −∇V, field lines.

### Phase 4 — Potential flow
`src/lib/pde/potentialFlow.ts` + `PotentialFlowPanel.tsx`. Velocity potential, streamlines, Bernoulli pressure, adaptive refinement reusing `lib/refinement`.

### Phase 5 — Structural potential
`src/lib/pde/structuralPotential.ts`. Airy stress / harmonic deformation smoothing, crack-risk indicator feeding `lib/topology`.

### Phase 6 — Harmonic robotics
`src/lib/pde/navField.ts` + `NavFieldPanel.tsx`. Harmonic potential w/ obstacles as Dirichlet sinks, ∇V descent for paths.

### Phase 7 — Reduced-order physics
`src/lib/pde/rom.ts`. POD/Krylov modal basis from snapshots, Galerkin projection, instant-preview surrogate.

### Phase 8 — Differentiable PDE
`src/lib/pde/diff.ts`. Adjoint solver for ∂L/∂κ, ∂L/∂BC, ∂L/∂geometry; plugs into existing `sdf/differentiable.ts` and `materialModel.ts` gradients.

### Phase 9 — Geometry OS integration & dashboard
- Spectral Laplacian shared with `lib/topology/embeddings`.
- New `/physics/fields` tab grouping thermal/electro/flow/nav/structural panels with one shared mesh + solver pipeline.
- GPU path via `lib/sdf/gpu` + new `pde/gpu.ts` (WebGPU CG kernel, CPU fallback).

### Cross-cutting
- All solvers behind `createServerFn` (`src/lib/pde/*.functions.ts`); gated under existing `_physics` admin/role layout.
- GPU optional, CPU always works.
- Determinism + tests per phase (`*.test.ts`), `bunx tsc --noEmit` clean each turn.
- No new edge functions, no new deps unless strictly required.

### This turn
On approval I'll implement **Phase 1** only (core sparse + Laplacian + Poisson + CG + multigrid + tests) and stop, so you can verify before Phase 2.
