-- API clients (Fabrication OS, Midwater, etc.)
create table public.api_clients (
  id uuid primary key default gen_random_uuid(),
  name text not null unique,
  key_prefix text not null,
  key_hash text not null unique,
  scopes text[] not null default '{}',
  created_by uuid,
  created_at timestamptz not null default now(),
  last_used_at timestamptz,
  revoked_at timestamptz
);
alter table public.api_clients enable row level security;
-- No public RLS policies: only service-role (admin) can touch this table.
-- Authenticated app users (you) read/write via server functions using supabaseAdmin.

-- Per-request audit log
create table public.api_request_log (
  id bigserial primary key,
  client_id uuid references public.api_clients(id) on delete set null,
  route text not null,
  method text not null,
  status int not null,
  latency_ms int,
  created_at timestamptz not null default now()
);
alter table public.api_request_log enable row level security;
create index api_request_log_client_idx on public.api_request_log(client_id, created_at desc);

-- STEP analysis jobs
create table public.step_jobs (
  id uuid primary key default gen_random_uuid(),
  client_id uuid references public.api_clients(id) on delete set null,
  filename text not null,
  storage_path text not null,
  status text not null default 'queued', -- queued|parsing|reasoning|done|failed
  error text,
  geometry jsonb,
  reasoning jsonb,
  created_at timestamptz not null default now(),
  completed_at timestamptz
);
alter table public.step_jobs enable row level security;
create index step_jobs_status_idx on public.step_jobs(status, created_at desc);
create index step_jobs_client_idx on public.step_jobs(client_id, created_at desc);

-- Private storage bucket for uploaded STEP files
insert into storage.buckets (id, name, public) values ('step-uploads', 'step-uploads', false)
on conflict (id) do nothing;
-- No storage policies = only service role can read/write. That is what we want.