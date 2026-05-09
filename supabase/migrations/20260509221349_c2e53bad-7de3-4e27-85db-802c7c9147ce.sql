
-- Progress snapshot on the job row itself
alter table public.step_jobs
  add column if not exists progress jsonb;

-- Per-job event log
create table if not exists public.step_job_events (
  id bigserial primary key,
  job_id uuid not null references public.step_jobs(id) on delete cascade,
  stage text not null,
  progress integer not null default 0 check (progress between 0 and 100),
  message text,
  data jsonb,
  created_at timestamptz not null default now()
);

create index if not exists step_job_events_job_id_idx
  on public.step_job_events (job_id, id);

alter table public.step_job_events enable row level security;
-- No public policies; only service role (admin client) reads/writes.

-- Enable realtime so SSE route can subscribe via supabase channels.
alter table public.step_job_events replica identity full;
do $$
begin
  if not exists (
    select 1 from pg_publication_tables
    where pubname = 'supabase_realtime' and tablename = 'step_job_events'
  ) then
    execute 'alter publication supabase_realtime add table public.step_job_events';
  end if;
end $$;
