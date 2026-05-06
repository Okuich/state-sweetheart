// Domain Template Panel — pick template, tweak knobs, swap materials,
// preview the compiled SimParams patch, then apply it.

import { useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import {
  TEMPLATES, MATERIALS, getTemplate, compileTemplate,
  defaultKnobs, defaultMaterials, getMaterial,
  type KnobValues, type Domain,
} from "@/lib/templates";
import type { SimParams } from "@/components/PhysicsCanvas";

const DOMAIN_COLOR: Record<Domain, string> = {
  fabrication: "text-primary",
  manufacturing: "text-secondary",
  thermal: "text-destructive",
  stress: "text-accent",
  flow: "text-primary/80",
};

export function TemplatesPanel({
  onApply,
}: {
  onApply: (patch: Partial<SimParams>) => void;
}) {
  const [tmplId, setTmplId] = useState(TEMPLATES[0].id);
  const tmpl = useMemo(() => getTemplate(tmplId), [tmplId]);
  const [knobs, setKnobs] = useState<KnobValues>(() => defaultKnobs(tmpl));
  const [mats, setMats] = useState<Record<string, string>>(() => defaultMaterials(tmpl));

  // Reset knobs/materials when template switches.
  useEffect(() => {
    setKnobs(defaultKnobs(tmpl));
    setMats(defaultMaterials(tmpl));
  }, [tmpl]);

  const compiled = useMemo(() => compileTemplate(tmpl, knobs, mats), [tmpl, knobs, mats]);

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        <div>
          <div className="text-[10px] uppercase tracking-[0.3em] text-muted-foreground">
            templates/ — high-level domain abstractions · auto solver config
          </div>
          <div className="text-sm font-display mt-1">
            <span className={DOMAIN_COLOR[tmpl.domain]}>{tmpl.domain}</span> · {tmpl.label}
          </div>
        </div>
        <Button
          onClick={() => onApply(compiled)}
          className="bg-primary text-primary-foreground hover:bg-primary/90 uppercase tracking-[0.18em] text-xs"
        >
          apply template
        </Button>
      </div>

      {/* Template gallery */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-2 mb-4">
        {TEMPLATES.map((t) => (
          <button
            key={t.id}
            onClick={() => setTmplId(t.id)}
            className={`text-left rounded-md border p-2 transition ${
              t.id === tmplId
                ? "border-primary/60 bg-primary/10"
                : "border-border bg-background/40 hover:bg-background/60"
            }`}
          >
            <div className={`text-[9px] uppercase tracking-[0.2em] ${DOMAIN_COLOR[t.domain]}`}>{t.domain}</div>
            <div className="text-xs font-medium mt-0.5">{t.label}</div>
          </button>
        ))}
      </div>

      <div className="text-xs text-muted-foreground mb-4 leading-relaxed">{tmpl.description}</div>

      <div className="grid lg:grid-cols-2 gap-4">
        {/* Knobs */}
        <div className="rounded-md border border-border bg-background/40 p-3">
          <div className="text-[10px] uppercase tracking-[0.3em] text-muted-foreground mb-3">
            parameters
          </div>
          <div className="space-y-3">
            {tmpl.knobs.map((k) => {
              if (k.kind === "select") {
                const v = (knobs[k.id] as string) ?? k.default;
                return (
                  <label key={k.id} className="block text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
                    {k.label}
                    <select
                      value={v}
                      onChange={(e) => setKnobs((s) => ({ ...s, [k.id]: e.target.value }))}
                      className="mt-1 w-full bg-background border border-border rounded-sm px-2 py-1 text-xs text-foreground"
                    >
                      {k.options.map((o) => (
                        <option key={o.value} value={o.value}>{o.label}</option>
                      ))}
                    </select>
                  </label>
                );
              }
              const v = (knobs[k.id] as number) ?? k.default;
              return (
                <div key={k.id} className="space-y-1.5">
                  <div className="flex justify-between text-[10px] uppercase tracking-[0.18em]">
                    <span className="text-muted-foreground">{k.label}</span>
                    <span className="text-primary tabular-nums">
                      {v.toFixed(k.step < 1 ? 1 : 0)}
                      {k.unit && <span className="text-muted-foreground ml-1">{k.unit}</span>}
                    </span>
                  </div>
                  <Slider
                    value={[v]} min={k.min} max={k.max} step={k.step}
                    onValueChange={([x]) => setKnobs((s) => ({ ...s, [k.id]: x }))}
                  />
                </div>
              );
            })}
          </div>

          {/* Material slots */}
          <div className="mt-4 pt-3 border-t border-border">
            <div className="text-[10px] uppercase tracking-[0.3em] text-muted-foreground mb-2">
              material library
            </div>
            <div className="space-y-2">
              {tmpl.materialSlots.map((slot) => {
                const m = getMaterial(mats[slot.id]);
                return (
                  <label key={slot.id} className="block text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
                    <div className="flex items-center justify-between">
                      <span>{slot.label}</span>
                      <span className="flex items-center gap-1.5 text-foreground/80 normal-case tracking-normal text-[11px]">
                        <span
                          className="inline-block h-2.5 w-2.5 rounded-sm border border-border"
                          style={{ background: m.color }}
                        />
                        ρ {m.density} · E {m.young} GPa
                      </span>
                    </div>
                    <select
                      value={mats[slot.id]}
                      onChange={(e) => setMats((s) => ({ ...s, [slot.id]: e.target.value }))}
                      className="mt-1 w-full bg-background border border-border rounded-sm px-2 py-1 text-xs text-foreground"
                    >
                      {MATERIALS.map((mm) => (
                        <option key={mm.id} value={mm.id}>{mm.label}</option>
                      ))}
                    </select>
                  </label>
                );
              })}
            </div>
          </div>
        </div>

        {/* Compiled output preview */}
        <div className="rounded-md border border-border bg-background/40 p-3">
          <div className="text-[10px] uppercase tracking-[0.3em] text-muted-foreground mb-2">
            compiled SimParams patch · solver={tmpl.solver.integrator}/{tmpl.solver.pairwiseAlgo}
          </div>
          <pre className="text-[11px] font-mono text-foreground/90 leading-relaxed overflow-auto max-h-80 bg-background/60 rounded-sm p-2">
{JSON.stringify(compiled, null, 2)}
          </pre>
          <div className="mt-2 text-[10px] text-muted-foreground">
            Click <span className="text-primary uppercase tracking-[0.18em]">apply template</span> to push these
            values into the live simulation.
          </div>
        </div>
      </div>
    </div>
  );
}
