
-- Phase 1: Roles + Feature flags + Role requests
-- ─────────────────────────────────────────────────────────

-- Roles
DO $$ BEGIN
  CREATE TYPE public.app_role AS ENUM ('user', 'enterprise', 'admin');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

CREATE TABLE IF NOT EXISTS public.user_roles (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  role public.app_role NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (user_id, role)
);

ALTER TABLE public.user_roles ENABLE ROW LEVEL SECURITY;

-- Security-definer role check (avoids recursive RLS)
CREATE OR REPLACE FUNCTION public.has_role(_user_id uuid, _role public.app_role)
RETURNS boolean
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT EXISTS (
    SELECT 1 FROM public.user_roles
    WHERE user_id = _user_id AND role = _role
  );
$$;

-- Helper: is the caller an admin?
CREATE OR REPLACE FUNCTION public.is_admin()
RETURNS boolean
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT public.has_role(auth.uid(), 'admin'::public.app_role);
$$;

-- user_roles policies
DROP POLICY IF EXISTS "user_roles read own" ON public.user_roles;
CREATE POLICY "user_roles read own" ON public.user_roles
  FOR SELECT USING (auth.uid() = user_id OR public.is_admin());

DROP POLICY IF EXISTS "user_roles admin write" ON public.user_roles;
CREATE POLICY "user_roles admin write" ON public.user_roles
  FOR ALL USING (public.is_admin()) WITH CHECK (public.is_admin());

-- ─────────────────────────────────────────────────────────
-- Feature flags
CREATE TABLE IF NOT EXISTS public.feature_flags (
  key text PRIMARY KEY,
  description text,
  enabled boolean NOT NULL DEFAULT false,
  allowed_roles text[] NOT NULL DEFAULT '{}',
  rollout_percentage integer NOT NULL DEFAULT 0 CHECK (rollout_percentage BETWEEN 0 AND 100),
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE public.feature_flags ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "feature_flags public read" ON public.feature_flags;
CREATE POLICY "feature_flags public read" ON public.feature_flags
  FOR SELECT USING (true);

DROP POLICY IF EXISTS "feature_flags admin write" ON public.feature_flags;
CREATE POLICY "feature_flags admin write" ON public.feature_flags
  FOR ALL USING (public.is_admin()) WITH CHECK (public.is_admin());

CREATE TRIGGER feature_flags_set_updated_at
  BEFORE UPDATE ON public.feature_flags
  FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

-- Per-user overrides
CREATE TABLE IF NOT EXISTS public.user_feature_flags (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  flag_key text NOT NULL REFERENCES public.feature_flags(key) ON DELETE CASCADE,
  enabled boolean NOT NULL,
  granted_by uuid REFERENCES auth.users(id),
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (user_id, flag_key)
);

ALTER TABLE public.user_feature_flags ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "user_feature_flags read own" ON public.user_feature_flags;
CREATE POLICY "user_feature_flags read own" ON public.user_feature_flags
  FOR SELECT USING (auth.uid() = user_id OR public.is_admin());

DROP POLICY IF EXISTS "user_feature_flags admin write" ON public.user_feature_flags;
CREATE POLICY "user_feature_flags admin write" ON public.user_feature_flags
  FOR ALL USING (public.is_admin()) WITH CHECK (public.is_admin());

-- ─────────────────────────────────────────────────────────
-- Role / access requests
CREATE TABLE IF NOT EXISTS public.role_requests (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  requested_role public.app_role NOT NULL DEFAULT 'enterprise',
  requested_flag text,
  reason text,
  status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','approved','rejected')),
  reviewed_by uuid REFERENCES auth.users(id),
  reviewed_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE public.role_requests ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "role_requests read own" ON public.role_requests;
CREATE POLICY "role_requests read own" ON public.role_requests
  FOR SELECT USING (auth.uid() = user_id OR public.is_admin());

DROP POLICY IF EXISTS "role_requests insert own" ON public.role_requests;
CREATE POLICY "role_requests insert own" ON public.role_requests
  FOR INSERT WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "role_requests admin update" ON public.role_requests;
CREATE POLICY "role_requests admin update" ON public.role_requests
  FOR UPDATE USING (public.is_admin()) WITH CHECK (public.is_admin());

CREATE TRIGGER role_requests_set_updated_at
  BEFORE UPDATE ON public.role_requests
  FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

-- ─────────────────────────────────────────────────────────
-- Physics analysis jobs (port of physicsClient artifacts)
CREATE TABLE IF NOT EXISTS public.physics_jobs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  kind text NOT NULL,                  -- 'analyze' | 'batch' | 'pipeline' | 'compare'
  status text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued','running','done','failed')),
  input jsonb NOT NULL,
  result jsonb,
  error text,
  duration_ms integer,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE public.physics_jobs ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "physics_jobs read own" ON public.physics_jobs;
CREATE POLICY "physics_jobs read own" ON public.physics_jobs
  FOR SELECT USING (auth.uid() = user_id OR public.is_admin());

DROP POLICY IF EXISTS "physics_jobs insert own" ON public.physics_jobs;
CREATE POLICY "physics_jobs insert own" ON public.physics_jobs
  FOR INSERT WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "physics_jobs update own" ON public.physics_jobs;
CREATE POLICY "physics_jobs update own" ON public.physics_jobs
  FOR UPDATE USING (auth.uid() = user_id OR public.is_admin());

CREATE INDEX IF NOT EXISTS physics_jobs_user_created_idx
  ON public.physics_jobs (user_id, created_at DESC);

CREATE TRIGGER physics_jobs_set_updated_at
  BEFORE UPDATE ON public.physics_jobs
  FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

-- ─────────────────────────────────────────────────────────
-- Materials catalog (writable by admins, public read)
CREATE TABLE IF NOT EXISTS public.materials (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  slug text NOT NULL UNIQUE,
  name text NOT NULL,
  family text NOT NULL,                -- 'metal' | 'polymer' | 'ceramic' | 'composite'
  density_kg_m3 numeric,
  youngs_modulus_gpa numeric,
  yield_strength_mpa numeric,
  ultimate_strength_mpa numeric,
  thermal_conductivity_w_mk numeric,
  cost_usd_per_kg numeric,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE public.materials ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "materials public read" ON public.materials;
CREATE POLICY "materials public read" ON public.materials FOR SELECT USING (true);

DROP POLICY IF EXISTS "materials admin write" ON public.materials;
CREATE POLICY "materials admin write" ON public.materials
  FOR ALL USING (public.is_admin()) WITH CHECK (public.is_admin());

CREATE TRIGGER materials_set_updated_at
  BEFORE UPDATE ON public.materials
  FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

-- ─────────────────────────────────────────────────────────
-- Patent records (per-user IP suite)
CREATE TABLE IF NOT EXISTS public.patent_records (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  title text NOT NULL,
  status text NOT NULL DEFAULT 'draft',
  jurisdiction text,
  claims jsonb NOT NULL DEFAULT '[]'::jsonb,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE public.patent_records ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "patent_records read own" ON public.patent_records;
CREATE POLICY "patent_records read own" ON public.patent_records
  FOR SELECT USING (auth.uid() = user_id OR public.is_admin());

DROP POLICY IF EXISTS "patent_records write own" ON public.patent_records;
CREATE POLICY "patent_records write own" ON public.patent_records
  FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

CREATE TRIGGER patent_records_set_updated_at
  BEFORE UPDATE ON public.patent_records
  FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

-- ─────────────────────────────────────────────────────────
-- Pilot engagements (business surface)
CREATE TABLE IF NOT EXISTS public.pilot_engagements (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  company text NOT NULL,
  contact_email text NOT NULL,
  stage text NOT NULL DEFAULT 'inquiry',
  notes text,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE public.pilot_engagements ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "pilot_engagements read own" ON public.pilot_engagements;
CREATE POLICY "pilot_engagements read own" ON public.pilot_engagements
  FOR SELECT USING (auth.uid() = user_id OR public.is_admin());

DROP POLICY IF EXISTS "pilot_engagements insert own" ON public.pilot_engagements;
CREATE POLICY "pilot_engagements insert own" ON public.pilot_engagements
  FOR INSERT WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "pilot_engagements update own or admin" ON public.pilot_engagements;
CREATE POLICY "pilot_engagements update own or admin" ON public.pilot_engagements
  FOR UPDATE USING (auth.uid() = user_id OR public.is_admin());

CREATE TRIGGER pilot_engagements_set_updated_at
  BEFORE UPDATE ON public.pilot_engagements
  FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

-- ─────────────────────────────────────────────────────────
-- Seed feature flags
INSERT INTO public.feature_flags (key, description, enabled, allowed_roles, rollout_percentage)
VALUES
  ('physics_engine', 'Access to advanced physics analysis (analyze, batch, pipeline, compare)', true, ARRAY['enterprise','admin'], 0),
  ('ml_training', 'Access to ML training dashboard and model evaluation', true, ARRAY['enterprise','admin'], 0),
  ('patent_suite', 'Access to patent / IP analysis tools', true, ARRAY['enterprise','admin'], 0),
  ('materials_catalog_edit', 'Edit materials catalog entries', true, ARRAY['admin'], 0),
  ('admin_panel', 'Access to admin panel and role requests', true, ARRAY['admin'], 0)
ON CONFLICT (key) DO NOTHING;
