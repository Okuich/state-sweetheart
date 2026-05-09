/**
 * Server functions for the internal STEP jobs dashboard.
 */
import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";
import { requireSupabaseAuth } from "@/integrations/supabase/auth-middleware";
import { supabaseAdmin } from "@/integrations/supabase/client.server";

const STATUSES = ["queued", "parsing", "reasoning", "done", "failed"] as const;

export const listStepJobs = createServerFn({ method: "GET" })
  .middleware([requireSupabaseAuth])
  .inputValidator((d) =>
    z
      .object({
        clientId: z.string().uuid().optional(),
        status: z.enum(STATUSES).optional(),
        q: z.string().trim().max(200).optional(),
      })
      .optional()
      .parse(d ?? {}),
  )
  .handler(async ({ data }) => {
    let query = supabaseAdmin
      .from("step_jobs")
      .select("id,client_id,filename,status,error,created_at,completed_at")
      .order("created_at", { ascending: false })
      .limit(100);

    if (data?.clientId) query = query.eq("client_id", data.clientId);
    if (data?.status) query = query.eq("status", data.status);
    if (data?.q) query = query.ilike("filename", `%${data.q}%`);

    const { data: jobs, error } = await query;
    if (error) throw new Error(error.message);

    const { data: clients } = await supabaseAdmin
      .from("api_clients")
      .select("id,name")
      .order("name", { ascending: true });
    const nameById = new Map((clients ?? []).map((c) => [c.id, c.name]));

    return {
      jobs: (jobs ?? []).map((j) => ({
        ...j,
        client_name: j.client_id ? nameById.get(j.client_id) ?? null : null,
      })),
      clients: clients ?? [],
      statuses: STATUSES,
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
