/**
 * Helpers to emit progress events for STEP jobs.
 * Server-only. Writes to step_job_events (broadcast via realtime) and updates
 * the step_jobs.progress snapshot.
 */


export interface JobEvent {
  stage: string;
  progress: number; // 0-100
  message?: string;
  data?: unknown;
}

export async function emitJobEvent(jobId: string, ev: JobEvent): Promise<void> {
  const progress = Math.max(0, Math.min(100, Math.round(ev.progress)));
  const row = {
    job_id: jobId,
    stage: ev.stage,
    progress,
    message: ev.message ?? null,
    data: (ev.data ?? null) as never,
  };
  // Insert event (fan-out via realtime) and update snapshot.
  await Promise.all([
    supabaseAdmin.from("step_job_events").insert(row),
    supabaseAdmin
      .from("step_jobs")
      .update({
        progress: {
          stage: ev.stage,
          progress,
          message: ev.message ?? null,
          at: new Date().toISOString(),
        } as never,
      })
      .eq("id", jobId),
  ]);
}
