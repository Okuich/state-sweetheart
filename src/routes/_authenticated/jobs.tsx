import { createFileRoute, Link } from "@tanstack/react-router";
import { useServerFn } from "@tanstack/react-start";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { listStepJobs } from "@/lib/step-jobs.functions";

export const Route = createFileRoute("/_authenticated/jobs")({
  component: JobsPage,
  head: () => ({ meta: [{ title: "STEP Jobs — Particle Dynamics Engine" }] }),
});

function statusVariant(s: string): "default" | "destructive" | "secondary" {
  if (s === "failed") return "destructive";
  if (s === "done") return "default";
  return "secondary";
}

function JobsPage() {
  const list = useServerFn(listStepJobs);
  const { data, isLoading, refetch } = useQuery({
    queryKey: ["step-jobs"],
    queryFn: () => list(),
    refetchInterval: 5000,
  });

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="mx-auto max-w-5xl space-y-6">
        <header className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">STEP Jobs</h1>
            <p className="text-sm text-muted-foreground">
              Recent uploads from Fabrication OS / Midwater. Auto-refreshes every 5 s.
            </p>
          </div>
          <div className="flex gap-3 text-sm">
            <Link to="/api-keys" className="text-muted-foreground underline">
              API Keys
            </Link>
            <Link to="/" className="text-muted-foreground underline">
              ← Engine
            </Link>
          </div>
        </header>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>Recent (last 100)</CardTitle>
            <button
              className="text-xs text-muted-foreground underline"
              onClick={() => refetch()}
            >
              Refresh
            </button>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <p className="text-sm text-muted-foreground">Loading…</p>
            ) : (
              <table className="w-full text-sm">
                <thead className="text-xs text-muted-foreground">
                  <tr className="text-left">
                    <th className="py-2">Filename</th>
                    <th>Caller</th>
                    <th>Status</th>
                    <th>Created</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {(data?.jobs ?? []).map((j) => (
                    <tr key={j.id} className="border-t border-border">
                      <td className="py-2 font-mono text-xs">{j.filename}</td>
                      <td className="text-xs">{j.client_name ?? "—"}</td>
                      <td>
                        <Badge variant={statusVariant(j.status)}>{j.status}</Badge>
                      </td>
                      <td className="text-xs">
                        {new Date(j.created_at).toLocaleString()}
                      </td>
                      <td className="text-right">
                        <Link
                          to="/jobs/$id"
                          params={{ id: j.id }}
                          className="text-xs text-primary underline"
                        >
                          Open
                        </Link>
                      </td>
                    </tr>
                  ))}
                  {(data?.jobs ?? []).length === 0 && (
                    <tr>
                      <td colSpan={5} className="py-6 text-center text-xs text-muted-foreground">
                        No jobs yet.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
