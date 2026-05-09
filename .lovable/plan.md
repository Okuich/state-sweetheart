# Particle Dynamics Engine — Hidden Reasoning Layer

Goal: PDE becomes a headless service that Fabrication OS and Midwater call with their own API keys to upload STEP files and receive AI-reasoned analysis. You remain the only human-facing user (existing email/password login is preserved for the internal dashboard).

---

## 1. Service authentication (per-caller API keys)

**New table `api_clients`** (admin-managed, no public RLS):
- `name` (e.g. "fabrication-os", "midwater")
- `key_prefix` (first 8 chars, shown in UI for identification)
- `key_hash` (SHA-256 of full key — raw key only shown once at creation)
- `scopes` (text[] — e.g. `['step:ingest', 'step:read']`)
- `last_used_at`, `revoked_at`, `created_by`

**New table `api_request_log`** for audit (client_id, route, status, latency, timestamp).

**Middleware** `src/lib/service-auth.ts`:
- Reads `Authorization: Bearer pde_<key>` from request
- Hashes incoming key, looks up `api_clients` with `supabaseAdmin`
- Rejects if revoked, missing scope, or no match (timing-safe)
- Writes to `api_request_log`
- Applied to all `/api/public/step/*` routes

**Internal admin UI** at `/_authenticated/api-keys`:
- List clients, create new key (shows raw key once), revoke
- View recent request log per client

---

## 2. STEP file ingestion

**Storage bucket** `step-uploads` (private, service-role writes).

**Route** `POST /api/public/step/analyze`:
- Service-auth middleware
- Accepts multipart `.step` / `.stp` (up to 25 MB)
- Saves to storage, creates `step_jobs` row (status=`queued`)
- Returns `{ job_id }` immediately

**Route** `GET /api/public/step/jobs/:id`:
- Returns job status + extracted data + reasoning when ready

---

## 3. STEP parsing + reasoning pipeline

Server function `processStepJob(jobId)` triggered after upload:

**Stage A — Parse** (`occt-import-js` WASM, runs in Worker):
- Bounding box, volume, surface area, units
- Face/edge/vertex counts, solid count
- Detected features (holes, pockets, fillets via topology heuristics)
- Material hints if present in STEP header

**Stage B — Reason** (Lovable AI Gateway, `google/gemini-2.5-pro`):
- Feeds extracted geometry JSON to model
- Returns structured analysis: manufacturability notes, suggested tolerances, fixturing concerns, simulation parameter recommendations (ties into existing `recommendSimulationParameters`)
- Stored as JSONB on `step_jobs.reasoning`

**`step_jobs` table:**
- `client_id` (fk api_clients), `filename`, `storage_path`
- `status` (queued/parsing/reasoning/done/failed), `error`
- `geometry` jsonb, `reasoning` jsonb
- `created_at`, `completed_at`

---

## 4. Internal dashboard updates

Under existing `/_authenticated`:
- `/api-keys` — manage Fabrication OS / Midwater keys
- `/jobs` — view all STEP jobs across clients (filter by client, status)
- `/jobs/$id` — single job: download STEP, view geometry JSON, view AI reasoning

---

## Technical notes

- `occt-import-js` is WASM-based and Worker-compatible; bundles cleanly with Vite. Fallback if it fails: minimal STEP header parser for units + bounding-box-from-vertices.
- All `/api/public/step/*` routes bypass the existing `requireSupabaseAuth` (your login flow) — they use only the API-key middleware.
- API keys are stored only as SHA-256 hashes; raw key shown once at creation, never recoverable.
- Request log keeps 90 days; older rows pruned via daily cron (pg_cron).

---

## Build order

1. DB migration: `api_clients`, `api_request_log`, `step_jobs`, storage bucket
2. Service-auth middleware + key generation helper
3. Admin UI: `/api-keys` (create/list/revoke)
4. STEP upload route + job row creation
5. WASM parser integration + processing pipeline
6. AI reasoning stage
7. Job status route + internal `/jobs` viewer
8. Smoke test with curl using a generated key

I'll execute step by step and check in after the DB migration + service auth land, since those are the load-bearing pieces.