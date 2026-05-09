create table public.telemetry_samples (
  id bigserial primary key,
  ingested_at timestamptz not null default now(),
  t double precision not null,
  energy_drift_pct double precision,
  constraint_l2 double precision,
  divergence_risk double precision,
  velocity_max double precision,
  nan_count integer,
  source text
);

create index telemetry_samples_source_ingested_idx
  on public.telemetry_samples (source, ingested_at desc);
create index telemetry_samples_ingested_idx
  on public.telemetry_samples (ingested_at desc);

alter table public.telemetry_samples enable row level security;

-- Anyone can read recent telemetry (read-only dashboards, public demos).
create policy "telemetry_samples public read"
  on public.telemetry_samples
  for select
  using (true);

-- Inserts only happen from the server (service role bypasses RLS),
-- so no INSERT policy is granted to anon/authenticated users.