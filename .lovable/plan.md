## Goal
Merge **Physics Playground** into **Physics OS** as one unified platform with a tabbed dashboard, role-gated features, and a TanStack server-fn backend (no external Express service).

## Scope (from your answers)
- Core engineering + materials
- ML / training suite
- Patent / IP suite
- Business pages (pricing, ROI, enterprise demo, pilot tracking, admin)
- Backend: reimplement Playground's `physicsClient` (`analyze`, `batchAnalyze`, `runPipeline`, `compareEngines`) as `createServerFn` handlers backed by the platform's existing `lib/` (meshing, refinement, GPU, topology, materials).
- Auth: port `user_roles` + feature-flag system; gate paid panels behind `physics_access`.
- IA: unified tabbed dashboard. Existing platform panels (Topology, Refinement, Geometry, Distributed, etc.) become tabs alongside Playground's Analyze, Materials, Patents, ML, etc.

## Phased delivery
Each phase = one chat turn so you can verify before the next.

### Phase 1 — Foundations (this turn)
- Inventory both codebases (file-by-file map of what to copy / merge / drop).
- Create the DB schema migration: `app_role` enum, `user_roles`, `feature_flags`, `physics_jobs`, `analysis_results`, `materials`, `patent_records`, `pilot_engagements`, `role_requests`. RLS + `has_role()` security-definer fn.
- Add `useUserRole` + `useFeatureFlags` hooks (server-fn backed).
- Add `_authenticated` layout + role guard (`_admin`, `_physics`).
- Wire Google + email/password auth (currently profiles exist but no login UI).

### Phase 2 — Backend port
- Move `physicsClient.analyze/batchAnalyze/runPipeline/compareEngines` into `src/lib/physics/*.functions.ts`, calling existing platform kernels.
- Port `materialEngine`, `recommendationEngine`, `supplierEngine`, `costEngine` as server-side modules.
- Port `unifiedPipeline`, `iterativeRefinement`, `paretoEngine`, `tradeoffMatrix`, `sensitivityJacobian`, `regulatoryCompliance`, `historicalPatternMatching`.

### Phase 3 — Engineering result UI
- Copy `SafetyGauge`, `DeflectionGauge`, `InteractiveMohrCircle`, `BeamDeflection3D`, `ForceVectorViz`, `LoadPathPanel`, `exportReport`.
- New `/analyze` tab wired to the new `analyze` server fn.

### Phase 4 — Materials tab
- Catalog UI, recommendation engine UI, supplier/cost panels, MaterialEditor.

### Phase 5 — ML / training tab
- `TrainingVizPanel`, `HPSearchVizPanel`, `ModelArchitecturePanel`, `ModelEvaluationDashboard`, `FeatureAttributionPanel`, `DatasetExplorationPanel`.

### Phase 6 — Patent / IP tab
- `PatentLandscapeViz`, `PatentCandidateMap`, `CompetitorClaimAnalysis`, `ClaimCoverageHeatmap`, `FTOCoveragePanel`, `ProsecutionActionItems`.

### Phase 7 — Business surface
- `/pricing`, `/enterprise-demo`, `/roi`, `/pilots`, `AdminRequestsPanel`, `PhysicsUpgradePrompt`, role-request flow.

### Phase 8 — IA unification
- Convert root `/` into the unified tabbed dashboard. Group: **Engine** (Topology, Refinement, Geometry, Distributed, GPU), **Analyze** (gauges, Mohr, beam), **Materials**, **ML**, **Patents**, **Ops** (telemetry, anomaly, agents), **Admin**.
- Single sidebar + breadcrumb. Old standalone routes become tab deep-links.

### Phase 9 — Cleanup & verification
- Delete dead Playground concepts not migrated (e.g. its Express client).
- Run `tsc --noEmit`, smoke-test each tab.
- Document so you can safely delete Physics Playground.

## Technical notes
- Backend: only TanStack server fns; admin client only inside `*.functions.ts` server-only paths.
- Roles: separate `user_roles` table (never on profiles). `has_role(uid, role)` security-definer.
- Feature flags: `feature_flags(key text pk, enabled bool, required_role app_role)` checked server-side.
- All Playground components that import Express / `physicsClient` get rewritten to `useServerFn`.
- Imports of `@/integrations/api/*` from Playground get replaced with the new server fns.
- No new edge functions.

## Confirm
Approve this and I'll start **Phase 1** in the next turn (DB migration + auth + role hooks + inventory).
