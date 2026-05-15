import { createFileRoute, Link } from "@tanstack/react-router";
import { useServerFn } from "@tanstack/react-start";
import { useQuery, useMutation } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import {
  analyzePhysics,
  batchAnalyze,
  compareEngines,
  runPipeline,
  optimizeDesign,
  validateEngine,
  listMaterials,
  listMyJobs,
} from "@/lib/physics.functions";
import { usePhysicsAccess } from "@/hooks/useAccess";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";

export const Route = createFileRoute("/_authenticated/physics")({
  component: PhysicsDashboard,
  head: () => ({
    meta: [
      { title: "Physics Engine — Admin" },
      { name: "description", content: "Analytical physics engine: stress, deflection, safety, optimization, and validation." },
    ],
  }),
});

interface AnalyzeForm {
  material: string;
  volume: number;
  surfaceArea: number;
  crossSectionalArea: number;
  momentOfInertia: number;
  beamLength: number;
  force: number;
  geometryType: string;
}

const DEFAULT_FORM: AnalyzeForm = {
  material: "steel",
  volume: 0.035,
  surfaceArea: 2.4,
  crossSectionalArea: 5.89e-3,
  momentOfInertia: 4.54e-5,
  beamLength: 6,
  force: 50000,
  geometryType: "beam",
};

function buildBody(f: AnalyzeForm) {
  return {
    geometry: { volume: f.volume, surfaceArea: f.surfaceArea },
    material: f.material,
    crossSectionalArea: f.crossSectionalArea,
    momentOfInertia: f.momentOfInertia,
    beamLength: f.beamLength,
    loadProfile: { force: f.force, direction: { x: 0, y: -1, z: 0 } },
    geometryType: f.geometryType,
  };
}

function PhysicsDashboard() {
  const { hasAccess, loading } = usePhysicsAccess();

  if (loading) {
    return <div className="flex min-h-screen items-center justify-center text-sm text-muted-foreground">Loading access…</div>;
  }
  if (!hasAccess) {
    return (
      <div className="mx-auto max-w-xl px-6 py-20 text-center space-y-4">
        <h1 className="text-2xl font-semibold">Physics Engine locked</h1>
        <p className="text-sm text-muted-foreground">
          The <code>physics_engine</code> feature flag is not enabled for your account. Request access from the admin panel.
        </p>
        <Button asChild variant="outline"><Link to="/">Back home</Link></Button>
      </div>
    );
  }
  return <Dashboard />;
}

function Dashboard() {
  const [form, setForm] = useState<AnalyzeForm>(DEFAULT_FORM);

  const fetchMaterials = useServerFn(listMaterials);
  const fetchJobs = useServerFn(listMyJobs);
  const analyzeFn = useServerFn(analyzePhysics);
  const batchFn = useServerFn(batchAnalyze);
  const compareFn = useServerFn(compareEngines);
  const pipelineFn = useServerFn(runPipeline);
  const optimizeFn = useServerFn(optimizeDesign);
  const validateFn = useServerFn(validateEngine);

  const materialsQ = useQuery({ queryKey: ["physics-materials"], queryFn: () => fetchMaterials() });
  const jobsQ = useQuery({ queryKey: ["physics-jobs"], queryFn: () => fetchJobs(), refetchInterval: 5000 });

  const analyzeM = useMutation({ mutationFn: () => analyzeFn({ data: buildBody(form) }) });
  const batchM = useMutation({ mutationFn: () => batchFn({ data: buildBody(form) }) });
  const compareM = useMutation({ mutationFn: () => compareFn({ data: buildBody(form) }) });
  const pipelineM = useMutation({ mutationFn: () => pipelineFn({ data: buildBody(form) }) });
  const validateM = useMutation({ mutationFn: () => validateFn() });
  const optimizeM = useMutation({
    mutationFn: () =>
      optimizeFn({
        data: {
          baseGeometry: {
            id: "current", name: `${form.geometryType} (form)`,
            area: form.surfaceArea, volume: form.volume, length: form.beamLength,
            momentOfInertia: form.momentOfInertia, crossSectionalArea: form.crossSectionalArea,
          },
          force: form.force, direction: { x: 0, y: -1, z: 0 },
          geometryType: form.geometryType, useAllGeometries: true,
        },
      }),
  });

  // Refresh job history after each mutation succeeds
  useEffect(() => {
    const subs = [analyzeM, batchM, compareM, pipelineM, optimizeM, validateM];
    if (subs.some((m) => m.isSuccess || m.isError)) jobsQ.refetch();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analyzeM.isSuccess, batchM.isSuccess, compareM.isSuccess, pipelineM.isSuccess, optimizeM.isSuccess, validateM.isSuccess]);

  const materialOptions = useMemo(() => materialsQ.data?.keys ?? ["steel", "aluminum", "titanium"], [materialsQ.data]);

  return (
    <div className="mx-auto max-w-7xl px-6 py-10 space-y-8">
      <header className="flex items-center justify-between">
        <div>
          <div className="text-xs uppercase tracking-[0.32em] text-muted-foreground">Admin · Physics Engine</div>
          <h1 className="text-3xl font-semibold mt-1">Analytical Mechanics Console</h1>
          <p className="text-sm text-muted-foreground mt-1">Stress, deflection, safety, cost, optimization, and benchmark validation.</p>
        </div>
        <Button variant="outline" asChild><Link to="/">Sandbox →</Link></Button>
      </header>

      <Card>
        <CardHeader><CardTitle className="text-base">Geometry & load</CardTitle></CardHeader>
        <CardContent className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Field label="Material">
            <Select value={form.material} onValueChange={(v) => setForm({ ...form, material: v })}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>{materialOptions.map((k) => <SelectItem key={k} value={k}>{k}</SelectItem>)}</SelectContent>
            </Select>
          </Field>
          <NumField label="Volume (m³)" value={form.volume} step={0.001} onChange={(v) => setForm({ ...form, volume: v })} />
          <NumField label="Surface area (m²)" value={form.surfaceArea} step={0.1} onChange={(v) => setForm({ ...form, surfaceArea: v })} />
          <NumField label="Cross-section (m²)" value={form.crossSectionalArea} step={1e-4} onChange={(v) => setForm({ ...form, crossSectionalArea: v })} />
          <NumField label="Moment of inertia (m⁴)" value={form.momentOfInertia} step={1e-6} onChange={(v) => setForm({ ...form, momentOfInertia: v })} />
          <NumField label="Beam length (m)" value={form.beamLength} step={0.1} onChange={(v) => setForm({ ...form, beamLength: v })} />
          <NumField label="Force (N)" value={form.force} step={1000} onChange={(v) => setForm({ ...form, force: v })} />
          <Field label="Geometry type">
            <Input value={form.geometryType} onChange={(e) => setForm({ ...form, geometryType: e.target.value })} />
          </Field>
        </CardContent>
      </Card>

      <Tabs defaultValue="analyze" className="space-y-4">
        <TabsList className="flex flex-wrap">
          <TabsTrigger value="analyze">Analyze</TabsTrigger>
          <TabsTrigger value="pipeline">Pipeline</TabsTrigger>
          <TabsTrigger value="batch">Batch (all materials)</TabsTrigger>
          <TabsTrigger value="compare">Compare engines</TabsTrigger>
          <TabsTrigger value="optimize">Optimize</TabsTrigger>
          <TabsTrigger value="validate">Validate benchmarks</TabsTrigger>
          <TabsTrigger value="jobs">Jobs ({jobsQ.data?.length ?? 0})</TabsTrigger>
        </TabsList>

        <TabsContent value="analyze">
          <ActionPanel title="Single analysis" mutation={analyzeM} onRun={() => analyzeM.mutate()}>
            {analyzeM.data && <AnalyzeResult d={analyzeM.data} />}
          </ActionPanel>
        </TabsContent>
        <TabsContent value="pipeline">
          <ActionPanel title="Full pipeline" mutation={pipelineM} onRun={() => pipelineM.mutate()}>
            {pipelineM.data && <Json data={pipelineM.data} />}
          </ActionPanel>
        </TabsContent>
        <TabsContent value="batch">
          <ActionPanel title="Batch analysis (every material)" mutation={batchM} onRun={() => batchM.mutate()}>
            {batchM.data && (
              <table className="w-full text-sm">
                <thead className="text-left text-xs uppercase tracking-wider text-muted-foreground">
                  <tr><th className="py-2">Material</th><th>Safety factor</th><th>Class</th><th>Von Mises (MPa)</th><th>Deflection (mm)</th><th>Cost</th></tr>
                </thead>
                <tbody>
                  {batchM.data.results.map((r) => (
                    <tr key={r.materialKey} className="border-t border-border/40">
                      <td className="py-2 font-medium">{r.materialName}</td>
                      <td>{r.safety.safetyFactor.toFixed(2)}</td>
                      <td><Badge variant="outline">{r.safety.classification}</Badge></td>
                      <td>{(r.stress.vonMises / 1e6).toFixed(2)}</td>
                      <td>{r.deflection.deflectionMm.toFixed(2)}</td>
                      <td>${r.cost.totalCost.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </ActionPanel>
        </TabsContent>
        <TabsContent value="compare">
          <ActionPanel title="Engine comparison (v1 vs ML)" mutation={compareM} onRun={() => compareM.mutate()}>
            {compareM.data && <Json data={compareM.data} />}
          </ActionPanel>
        </TabsContent>
        <TabsContent value="optimize">
          <ActionPanel title="Multi-material × scale optimization" mutation={optimizeM} onRun={() => optimizeM.mutate()}>
            {optimizeM.data && (
              <div className="space-y-3">
                <div className="text-sm">
                  Evaluated <b>{optimizeM.data.totalEvaluated}</b> · feasible <b>{optimizeM.data.feasibleCount}</b>
                </div>
                {optimizeM.data.optimal && (
                  <Card><CardHeader><CardTitle className="text-sm">Optimal candidate</CardTitle></CardHeader>
                    <CardContent className="text-sm"><Json data={optimizeM.data.optimal} /></CardContent>
                  </Card>
                )}
              </div>
            )}
          </ActionPanel>
        </TabsContent>
        <TabsContent value="validate">
          <ActionPanel title="Benchmark validation suite" mutation={validateM} onRun={() => validateM.mutate()}>
            {validateM.data && (
              <div className="space-y-3">
                <div className="text-sm font-medium">{validateM.data.summary} · {validateM.data.overallPassRate}%</div>
                {validateM.data.cases.map((c) => (
                  <Card key={c.benchmarkId}>
                    <CardHeader className="flex flex-row items-center justify-between">
                      <CardTitle className="text-sm">{c.benchmarkId} · {c.benchmarkName}</CardTitle>
                      <Badge variant={c.overallPass ? "default" : "destructive"}>{c.overallPass ? "PASS" : "FAIL"}</Badge>
                    </CardHeader>
                    <CardContent>
                      <table className="w-full text-xs">
                        <thead className="text-left text-muted-foreground">
                          <tr><th className="py-1">Metric</th><th>Expected</th><th>Actual</th><th>Error %</th></tr>
                        </thead>
                        <tbody>
                          {c.metrics.map((m) => (
                            <tr key={m.metric} className={m.withinTolerance ? "" : "text-destructive"}>
                              <td className="py-1">{m.metric}</td>
                              <td>{m.expected.toFixed(4)}</td>
                              <td>{m.actual.toFixed(4)}</td>
                              <td>{m.errorPct.toFixed(3)}%</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </ActionPanel>
        </TabsContent>
        <TabsContent value="jobs">
          <Card>
            <CardHeader><CardTitle className="text-base">Recent runs</CardTitle></CardHeader>
            <CardContent>
              {jobsQ.isLoading && <div className="text-sm text-muted-foreground">Loading…</div>}
              {!!jobsQ.data?.length && (
                <table className="w-full text-sm">
                  <thead className="text-left text-xs uppercase tracking-wider text-muted-foreground">
                    <tr><th className="py-2">When</th><th>Kind</th><th>Status</th><th>Duration</th><th>Error</th></tr>
                  </thead>
                  <tbody>
                    {jobsQ.data.map((j) => (
                      <tr key={j.id} className="border-t border-border/40">
                        <td className="py-2">{new Date(j.created_at!).toLocaleString()}</td>
                        <td>{j.kind}</td>
                        <td><Badge variant={j.status === "completed" ? "default" : "destructive"}>{j.status}</Badge></td>
                        <td>{j.duration_ms}ms</td>
                        <td className="text-destructive text-xs">{j.error ?? ""}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              {!jobsQ.isLoading && !jobsQ.data?.length && <div className="text-sm text-muted-foreground">No runs yet.</div>}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5">
      <div className="text-[11px] uppercase tracking-wider text-muted-foreground">{label}</div>
      {children}
    </div>
  );
}

function NumField({ label, value, step, onChange }: { label: string; value: number; step: number; onChange: (v: number) => void }) {
  return (
    <Field label={label}>
      <Input type="number" value={value} step={step}
        onChange={(e) => { const n = Number(e.target.value); if (!Number.isNaN(n)) onChange(n); }} />
    </Field>
  );
}

function ActionPanel({
  title, mutation, onRun, children,
}: {
  title: string;
  mutation: { isPending: boolean; isError: boolean; error: unknown };
  onRun: () => void;
  children?: React.ReactNode;
}) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base">{title}</CardTitle>
        <Button onClick={onRun} disabled={mutation.isPending}>
          {mutation.isPending ? "Running…" : "Run"}
        </Button>
      </CardHeader>
      <CardContent className="space-y-3">
        {mutation.isError && (
          <div className="text-sm text-destructive">{String((mutation.error as Error)?.message ?? mutation.error)}</div>
        )}
        {children}
      </CardContent>
    </Card>
  );
}

function AnalyzeResult({ d }: { d: NonNullable<Awaited<ReturnType<typeof analyzePhysics>>> }) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
      <Stat label="Material" value={d.material.name} />
      <Stat label="Safety factor" value={d.safety.safetyFactor.toFixed(2)} accent={d.safety.classification} />
      <Stat label="Von Mises" value={`${(d.stress.vonMises / 1e6).toFixed(2)} MPa`} />
      <Stat label="Deflection" value={`${d.deflection.deflectionMm.toFixed(2)} mm`} accent={d.deflection.classification} />
      <Stat label="Span ratio" value={`L/${Math.round(d.deflection.spanRatio)}`} />
      <Stat label="Cost" value={`$${d.cost.totalCost.toFixed(2)}`} />
      <Stat label="Confidence" value={`${(((d.confidence?.overall ?? 0) as number) * 100).toFixed(0)}%`} />
      <Stat label="Engine" value={d.engineSource} />
      {!!d.recommendations?.length && (
        <div className="col-span-full">
          <div className="text-xs uppercase tracking-wider text-muted-foreground mb-2">Recommendations</div>
          <ul className="space-y-1 text-sm">
            {d.recommendations.map((r, i) => (<li key={i}><b>{r.title}</b> — <span className="text-muted-foreground">{r.detail}</span></li>))}
          </ul>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div className="rounded-md border border-border/60 bg-card/40 p-3">
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className="text-base font-medium mt-1 tabular-nums">{value}</div>
      {accent && <Badge variant="outline" className="mt-2 text-[10px]">{accent}</Badge>}
    </div>
  );
}

function Json({ data }: { data: unknown }) {
  return (
    <pre className="overflow-auto text-xs bg-muted/30 p-3 rounded-md max-h-96">{JSON.stringify(data, null, 2)}</pre>
  );
}
