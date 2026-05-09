import { createFileRoute, Link } from "@tanstack/react-router";
import { useServerFn } from "@tanstack/react-start";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { getStepJob } from "@/lib/step-jobs.functions";

export const Route = createFileRoute("/_authenticated/jobs/$id")({
  component: JobDetailPage,
  head: () => ({ meta: [{ title: "STEP Job — Particle Dynamics Engine" }] }),
});

function JobDetailPage() {
  const { id } = Route.useParams();
  const get = useServerFn(getStepJob);
  const { data, isLoading } = useQuery({
    queryKey: ["step-job", id],
    queryFn: () => get({ data: { id } }),
    refetchInterval: (q) => {
      const s = q.state.data?.job?.status;
      return s === "done" || s === "failed" ? false : 3000;
    },
  });

  if (isLoading) {
    return <div className="min-h-screen p-6 text-sm text-muted-foreground">Loading…</div>;
  }
  const job = data?.job;
  if (!job) {
    return <div className="min-h-screen p-6 text-sm">Job not found.</div>;
  }

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="mx-auto max-w-5xl space-y-6">
        <header className="flex items-center justify-between">
          <div>
            <h1 className="font-mono text-lg">{job.filename}</h1>
            <p className="text-xs text-muted-foreground">
              {job.client_name ?? "unknown caller"} ·{" "}
              {new Date(job.created_at).toLocaleString()}
            </p>
          </div>
          <div className="flex items-center gap-3">
            <Badge>{job.status}</Badge>
            <Link to="/jobs" className="text-sm text-muted-foreground underline">
              ← All jobs
            </Link>
          </div>
        </header>

        {(() => {
          const p = job.progress as { stage?: string; progress?: number; message?: string; at?: string } | null;
          if (!p || job.status === "done" || job.status === "failed") return null;
          return (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">
                  {p.stage ?? job.status} — {p.progress ?? 0}%
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="h-2 w-full overflow-hidden rounded bg-muted">
                  <div
                    className="h-full bg-primary transition-all"
                    style={{ width: `${Math.min(100, Math.max(0, p.progress ?? 0))}%` }}
                  />
                </div>
                {p.message && <p className="text-xs text-muted-foreground">{p.message}</p>}
              </CardContent>
            </Card>
          );
        })()}

        {job.error && (
          <Card className="border-destructive/40">
            <CardHeader>
              <CardTitle className="text-destructive">Error</CardTitle>
            </CardHeader>
            <CardContent>
              <pre className="overflow-auto text-xs">{job.error}</pre>
            </CardContent>
          </Card>
        )}

        <Card>
          <CardHeader>
            <CardTitle>Geometry</CardTitle>
          </CardHeader>
          <CardContent>
            {job.geometry ? (
              <pre className="max-h-[480px] overflow-auto rounded bg-muted/40 p-3 text-[11px] leading-relaxed">
                {JSON.stringify(job.geometry, null, 2)}
              </pre>
            ) : (
              <p className="text-xs text-muted-foreground">Pending parsing…</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>AI Reasoning</CardTitle>
          </CardHeader>
          <CardContent>
            {job.reasoning ? (
              <pre className="max-h-[480px] overflow-auto rounded bg-muted/40 p-3 text-[11px] leading-relaxed">
                {JSON.stringify(job.reasoning, null, 2)}
              </pre>
            ) : (
              <p className="text-xs text-muted-foreground">Pending reasoning…</p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
