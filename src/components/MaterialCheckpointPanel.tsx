import { useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import {
  advance, checkpoint, hashState, initMaterialState, replay, restore,
  RollbackBuffer, type Checkpoint, type MaterialState, type StepInputs,
} from "@/lib/materialCheckpoint";

const E = 16;

function makeInputs(deps: number): StepInputs {
  return {
    depsTrial: Float32Array.from({ length: E }, (_, i) => deps * (1 + 0.05 * i)),
    dt: 0.01, yieldStress: 100, hardening: 50, Emod: 1000, tau: 0.05, eta: 10,
  };
}

export function MaterialCheckpointPanel() {
  const stateRef = useRef<MaterialState>(initMaterialState(E));
  const bufRef = useRef<RollbackBuffer>(new RollbackBuffer(32));
  const inputLogRef = useRef<StepInputs[]>([]);
  const [, force] = useState(0);
  const repaint = () => force((n) => n + 1);

  const [strain, setStrain] = useState(0.2);
  const [replayResult, setReplayResult] = useState<{ ok: boolean; live: number; replayed: number } | null>(null);

  const stepN = (n: number) => {
    for (let k = 0; k < n; k++) {
      const inp = makeInputs(strain);
      inputLogRef.current.push(inp);
      advance(stateRef.current, inp);
    }
    repaint();
  };

  const capture = () => {
    bufRef.current.push(checkpoint(stateRef.current));
    repaint();
  };

  const rollback = (cp: Checkpoint) => {
    restore(stateRef.current, cp);
    inputLogRef.current = inputLogRef.current.slice(0, cp.step);
    setReplayResult(null);
    repaint();
  };

  const reset = () => {
    stateRef.current = initMaterialState(E);
    bufRef.current.clear();
    inputLogRef.current = [];
    setReplayResult(null);
    repaint();
  };

  const verifyReplay = () => {
    const cp = bufRef.current.findAtOrBefore(0);
    if (!cp) { setReplayResult({ ok: false, live: 0, replayed: 0 }); return; }
    const tail = inputLogRef.current.slice(cp.step);
    const re = replay(cp, E, tail);
    const live = hashState(stateRef.current);
    setReplayResult({ ok: re.finalHash === live, live, replayed: re.finalHash });
  };

  const checkpoints = bufRef.current.toArray();
  const s = stateRef.current;

  // Aggregate diagnostics over elements.
  const stats = useMemo(() => {
    let alphaMax = 0, fp00Max = 1, svMax = 0;
    for (let e = 0; e < E; e++) {
      if (s.alpha[e] > alphaMax) alphaMax = s.alpha[e];
      if (s.Fp[e * 4] > fp00Max) fp00Max = s.Fp[e * 4];
      if (Math.abs(s.Sv[e * 3]) > svMax) svMax = Math.abs(s.Sv[e * 3]);
    }
    return { alphaMax, fp00Max, svMax };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s.step]);

  return (
    <div className="space-y-4 rounded-lg border border-border bg-card p-4">
      <div className="flex items-center justify-between">
        <h3 className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
          Material checkpoints (Fp, Sv, α)
        </h3>
        <Button variant="ghost" size="sm" onClick={reset} className="h-7 px-2 text-xs">
          reset
        </Button>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="space-y-3">
          <div className="space-y-1">
            <div className="flex justify-between text-[11px]">
              <span className="text-muted-foreground">trial strain Δε</span>
              <span className="tabular-nums">{strain.toFixed(2)}</span>
            </div>
            <Slider value={[strain]} min={0} max={0.6} step={0.01}
                    onValueChange={(v) => setStrain(v[0])} />
          </div>

          <div className="flex flex-wrap gap-1">
            <Button size="sm" onClick={() => stepN(1)} className="h-8 px-3 text-xs">+1 step</Button>
            <Button size="sm" onClick={() => stepN(10)} className="h-8 px-3 text-xs">+10</Button>
            <Button size="sm" variant="secondary" onClick={capture} className="h-8 px-3 text-xs">
              capture checkpoint
            </Button>
            <Button size="sm" variant="outline" onClick={verifyReplay} className="h-8 px-3 text-xs">
              verify replay
            </Button>
          </div>

          <div className="grid grid-cols-3 gap-2 rounded border border-border/60 bg-background/40 p-2 text-[11px]">
            <div>
              <div className="text-[9px] uppercase tracking-[0.18em] text-muted-foreground">step</div>
              <div className="tabular-nums">{s.step}</div>
            </div>
            <div>
              <div className="text-[9px] uppercase tracking-[0.18em] text-muted-foreground">α max</div>
              <div className="tabular-nums">{stats.alphaMax.toFixed(4)}</div>
            </div>
            <div>
              <div className="text-[9px] uppercase tracking-[0.18em] text-muted-foreground">Fp[0,0] max</div>
              <div className="tabular-nums">{stats.fp00Max.toFixed(4)}</div>
            </div>
            <div className="col-span-3">
              <div className="text-[9px] uppercase tracking-[0.18em] text-muted-foreground">|Sv| max</div>
              <div className="tabular-nums">{stats.svMax.toExponential(2)}</div>
            </div>
            <div className="col-span-3">
              <div className="text-[9px] uppercase tracking-[0.18em] text-muted-foreground">state hash</div>
              <div className="tabular-nums font-mono text-[10px]">
                0x{hashState(s).toString(16).padStart(8, "0")}
              </div>
            </div>
          </div>

          {replayResult && (
            <div className={`rounded border p-2 text-[11px] ${
              replayResult.ok
                ? "border-green-500/40 bg-green-500/10 text-green-400"
                : "border-destructive/40 bg-destructive/10 text-destructive"
            }`}>
              {replayResult.ok
                ? "✓ replay matches live state — bit-identical reproduction"
                : "✗ replay diverged — non-determinism detected"}
              <div className="mt-1 font-mono text-[10px] opacity-80">
                live   0x{replayResult.live.toString(16).padStart(8, "0")}<br />
                replay 0x{replayResult.replayed.toString(16).padStart(8, "0")}
              </div>
            </div>
          )}
        </div>

        <div className="rounded border border-border/60 bg-background/40 p-2">
          <div className="mb-2 text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
            checkpoint ring ({checkpoints.length}/{bufRef.current.capacity})
          </div>
          {checkpoints.length === 0 ? (
            <div className="text-[11px] text-muted-foreground italic">
              No checkpoints captured yet.
            </div>
          ) : (
            <div className="space-y-1 max-h-64 overflow-y-auto">
              {checkpoints.slice().reverse().map((cp, i) => (
                <div key={i} className="flex items-center justify-between rounded bg-background/40 px-2 py-1 text-[11px]">
                  <div className="font-mono">
                    step {cp.step.toString().padStart(4)} · 0x{cp.hash.toString(16).padStart(8, "0")}
                  </div>
                  <Button size="sm" variant="ghost" className="h-6 px-2 text-[10px]"
                          onClick={() => rollback(cp)}>
                    rollback
                  </Button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
