import { createFileRoute, Link } from "@tanstack/react-router";
import { useServerFn } from "@tanstack/react-start";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";
import {
  createApiClient,
  listApiClients,
  revokeApiClient,
  rotateApiClient,
} from "@/lib/api-clients.functions";

export const Route = createFileRoute("/_authenticated/api-keys")({
  component: ApiKeysPage,
  head: () => ({
    meta: [{ title: "API Keys — Particle Dynamics Engine" }],
  }),
});

const SCOPE_OPTIONS = [
  "step:ingest",
  "step:read",
  "telemetry:write",
  "telemetry:read",
  "reasoner:invoke",
] as const;

const PRESETS: Record<string, { name: string; scopes: string[] }> = {
  "fabrication-os": {
    name: "fabrication-os",
    scopes: ["step:ingest", "step:read", "reasoner:invoke", "telemetry:write"],
  },
  midwater: {
    name: "midwater",
    scopes: ["step:ingest", "step:read", "telemetry:read"],
  },
};

function ApiKeysPage() {
  const qc = useQueryClient();
  const list = useServerFn(listApiClients);
  const create = useServerFn(createApiClient);
  const revoke = useServerFn(revokeApiClient);
  const rotate = useServerFn(rotateApiClient);

  const { data, isLoading } = useQuery({
    queryKey: ["api-clients"],
    queryFn: () => list(),
  });

  const [name, setName] = useState("");
  const [scopes, setScopes] = useState<string[]>(["step:ingest", "step:read"]);
  const [revealed, setRevealed] = useState<{ name: string; key: string } | null>(null);

  const createMut = useMutation({
    mutationFn: (input: { name: string; scopes: string[] }) => create({ data: input }),
    onSuccess: (res) => {
      setRevealed({ name: res.client.name, key: res.raw_key });
      setName("");
      qc.invalidateQueries({ queryKey: ["api-clients"] });
      toast.success("API key created — copy it now, it won't be shown again");
    },
    onError: (e: Error) => toast.error(e.message),
  });

  const revokeMut = useMutation({
    mutationFn: (id: string) => revoke({ data: { id } }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["api-clients"] });
      toast.success("Revoked");
    },
    onError: (e: Error) => toast.error(e.message),
  });

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="mx-auto max-w-4xl space-y-6">
        <header className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">API Keys</h1>
            <p className="text-sm text-muted-foreground">
              Per-caller credentials for Fabrication OS, Midwater, and other services.
            </p>
          </div>
          <Link to="/" className="text-sm text-muted-foreground underline">
            ← Engine
          </Link>
        </header>

        <Card>
          <CardHeader>
            <CardTitle>Create new key</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-wrap gap-2">
              <span className="text-xs text-muted-foreground self-center">Presets:</span>
              {Object.entries(PRESETS).map(([key, p]) => (
                <Button
                  key={key}
                  variant="outline"
                  size="sm"
                  type="button"
                  onClick={() => {
                    setName(p.name);
                    setScopes(p.scopes);
                  }}
                >
                  {p.name}
                </Button>
              ))}
            </div>
            <div className="flex flex-wrap gap-3">
              <Input
                placeholder="caller name (e.g. fabrication-os)"
                value={name}
                onChange={(e) => setName(e.target.value.toLowerCase())}
                className="max-w-xs"
              />
              <div className="flex flex-wrap items-center gap-2">
                {SCOPE_OPTIONS.map((s) => (
                  <label key={s} className="flex items-center gap-1 text-xs">
                    <input
                      type="checkbox"
                      checked={scopes.includes(s)}
                      onChange={(e) =>
                        setScopes((prev) =>
                          e.target.checked ? [...prev, s] : prev.filter((x) => x !== s),
                        )
                      }
                    />
                    {s}
                  </label>
                ))}
              </div>
              <Button
                disabled={!name || scopes.length === 0 || createMut.isPending}
                onClick={() => createMut.mutate({ name, scopes })}
              >
                {createMut.isPending ? "Creating…" : "Create"}
              </Button>
            </div>

            {revealed && (
              <div className="rounded-md border border-amber-500/40 bg-amber-500/5 p-3 text-xs">
                <div className="mb-2 font-medium">
                  Copy the key for <code>{revealed.name}</code> now — it will not be shown again.
                </div>
                <code className="block break-all rounded bg-background px-2 py-1 font-mono">
                  {revealed.key}
                </code>
                <Button
                  size="sm"
                  variant="ghost"
                  className="mt-2"
                  onClick={() => {
                    navigator.clipboard.writeText(revealed.key);
                    toast.success("Copied");
                  }}
                >
                  Copy
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Active clients</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <p className="text-sm text-muted-foreground">Loading…</p>
            ) : (
              <table className="w-full text-sm">
                <thead className="text-xs text-muted-foreground">
                  <tr className="text-left">
                    <th className="py-2">Name</th>
                    <th>Prefix</th>
                    <th>Scopes</th>
                    <th>Last used</th>
                    <th>Status</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {(data?.clients ?? []).map((c) => (
                    <tr key={c.id} className="border-t border-border">
                      <td className="py-2 font-medium">{c.name}</td>
                      <td className="font-mono text-xs">{c.key_prefix}…</td>
                      <td className="text-xs">{c.scopes?.join(", ")}</td>
                      <td className="text-xs">
                        {c.last_used_at ? new Date(c.last_used_at).toLocaleString() : "—"}
                      </td>
                      <td>
                        {c.revoked_at ? (
                          <Badge variant="destructive">revoked</Badge>
                        ) : (
                          <Badge>active</Badge>
                        )}
                      </td>
                      <td className="text-right">
                        {!c.revoked_at && (
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => revokeMut.mutate(c.id)}
                          >
                            Revoke
                          </Button>
                        )}
                      </td>
                    </tr>
                  ))}
                  {(data?.clients ?? []).length === 0 && (
                    <tr>
                      <td colSpan={6} className="py-6 text-center text-xs text-muted-foreground">
                        No clients yet. Create one above.
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
