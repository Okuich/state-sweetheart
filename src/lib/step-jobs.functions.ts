/**
 * Server functions for the internal STEP jobs dashboard.
 */
import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";
import { requireSupabaseAuth } from "@/integrations/supabase/auth-middleware";
import { supabaseAdmin } from "@/integrations/supabase/client.server";

export const listStepJobs = createServerFn({ method: "GET" })
  .middleware([requireSupabaseAuth])
  .handler(async () => {
    const { data: jobs, error } = await supabaseAdmin
      .from("step_jobs")
      .select("id,client_id,filename,status,error,created_at,completed_at")
      .order("created_at", { ascending: false })
      .limit(100);
    if (error) throw new Error(error.message);

    const { data: clients } = await supabaseAdmin
      .from("api_clients")
      .select("id,name");
    const nameById = new Map((clients ?? []).map((c) => [c.id, c.name]));

    return {
      jobs: (jobs ?? []).map((j) => ({
        ...j,
        client_name: j.client_id ? nameById.get(j.client_id) ?? null : null,
      })),
    };
  });

export const getStepJob = createServerFn({ method: "GET" })
  .middleware([requireSupabaseAuth])
  .inputValidator((d) => z.object({ id: z.string().uuid() }).parse(d))
  .handler(async ({ data }) => {
    const { data: job, error } = await supabaseAdmin
      .from("step_jobs")
      .select("*")
      .eq("id", data.id)
      .maybeSingle();
    if (error) throw new Error(error.message);
    if (!job) throw new Error("not found");

    let client_name: string | null = null;
    if (job.client_id) {
      const { data: c } = await supabaseAdmin
        .from("api_clients")
        .select("name")
        .eq("id", job.client_id)
        .maybeSingle();
      client_name = c?.name ?? null;
    }
    return { job: { ...job, client_name } };
  });
