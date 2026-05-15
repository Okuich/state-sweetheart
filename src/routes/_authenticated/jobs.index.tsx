import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useServerFn } from "@tanstack/react-start";
import { useQuery } from "@tanstack/react-query";
import { zodValidator, fallback } from "@tanstack/zod-adapter";
import { z } from "zod";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { listStepJobs } from "@/lib/step-jobs.functions";

const STATUSES = ["queued", "parsing", "reasoning", "done", "failed"] as const;

const searchSchema = z.object({
  clientId: fallback(z.string().uuid().optional(), undefined),
  status: fallback(z.enum(STATUSES).optional(), undefined),
  q: fallback(z.string().optional(), undefined),
});

export const Route = createFileRoute("/_authenticated/jobs/")({
  validateSearch: zodValidator(searchSchema),
  component: JobsPage,
  head: () => ({
    meta: [
      { title: "STEP Jobs — Midwater" },
      { name: "description", content: "Midwater — STEP ingest jobs, callers, and status." },
    ],
  }),
});

function statusVariant(s: string): "default" | "destructive" | "secondary" {
  if (s === "failed") return "destructive";
  if (s === "done") return "default";
  return "secondary";
}

function JobsPage() {
  const { clientId, status, q } = Route.useSearch();
  const navigate = useNavigate({ from: "/jobs" });
  const list = useServerFn(listStepJobs);

  const { data, isLoading, refetch } = useQuery({
    queryKey: ["step-jobs", clientId ?? "", status ?? "", q ?? ""],
    queryFn: () => list({ data: { clientId, status, q } }),
    refetchInterval: 5000,
  });

  const setSearch = (next: Partial<{ clientId?: string; status?: string; q?: string }>) =>
    navigate({
      search: (prev: Record<string, unknown>) => ({
        ...prev,
        ...next,
        ...(next.clientId === "" ? { clientId: undefined } : {}),
        ...(next.status === "" ? { status: undefined } : {}),
        ...(next.q === "" ? { q: undefined } : {}),
      }),
    });

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="mx-auto max-w-6xl space-y-6">
        <header className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">STEP Jobs</h1>
            <p className="text-sm text-muted-foreground">
              Search by caller, status, or filename. Auto-refreshes every 5 s.
            </p>
          </div>
          <div className="flex gap-3 text-sm">
            <Link to="/api-keys" className="text-muted-foreground underline">
              API Keys
            </Link>
            <Link to="/" className="text-muted-foreground underline">
              ← Midwater
            </Link>
          </div>
        </header>

        <Card>
          <CardHeader className="space-y-3">
            <div className="flex items-center justify-between">
              <CardTitle>Recent (last 100)</CardTitle>
              <button
                className="text-xs text-muted-foreground underline"
                onClick={() => refetch()}
              >
                Refresh
              </button>
            </div>
            <div className="grid grid-cols-1 gap-2 md:grid-cols-[1fr_220px_180px_auto]">
              <Input
                placeholder="Search filename…"
                value={q ?? ""}
                onChange={(e) => setSearch({ q: e.target.value })}
                className="h-9"
              />
              <select
                className="h-9 rounded-md border border-input bg-background px-2 text-sm"
                value={clientId ?? ""}
                onChange={(e) => setSearch({ clientId: e.target.value || undefined })}
              >
                <option value="">All callers</option>
                {(data?.clients ?? []).map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.name}
                  </option>
                ))}
              </select>
              <select
                className="h-9 rounded-md border border-input bg-background px-2 text-sm"
                value={status ?? ""}
                onChange={(e) =>
                  setSearch({ status: (e.target.value || undefined) as typeof status })
                }
              >
                <option value="">Any status</option>
                {STATUSES.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
              {(clientId || status || q) && (
                <button
                  className="h-9 rounded-md border border-input px-3 text-xs text-muted-foreground hover:bg-muted"
                  onClick={() =>
                    navigate({ search: () => ({ clientId: undefined, status: undefined, q: undefined }) })
                  }
                >
                  Clear
                </button>
              )}
            </div>
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
                    <th>Error</th>
                    <th>Created</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {(data?.jobs ?? []).map((j) => (
                    <tr key={j.id} className="border-t border-border align-top">
                      <td className="py-2 font-mono text-xs">{j.filename}</td>
                      <td className="text-xs">{j.client_name ?? "—"}</td>
                      <td>
                        <Badge variant={statusVariant(j.status)}>{j.status}</Badge>
                      </td>
                      <td className="max-w-[280px] truncate text-xs text-destructive" title={j.error ?? ""}>
                        {j.error ?? ""}
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
                      <td colSpan={6} className="py-6 text-center text-xs text-muted-foreground">
                        No jobs match the current filters.
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
