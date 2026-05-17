import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import {
  FACE_KEYS, MODE_LABEL,
  type FaceBC, type FaceKey, type FaceMode,
} from "./boundary-types";

function NumField({
  label, value, step, onChange,
}: { label: string; value: number; step: number; onChange: (v: number) => void }) {
  return (
    <div className="space-y-1">
      <Label className="text-[11px] uppercase tracking-wide text-muted-foreground">{label}</Label>
      <Input
        type="number"
        value={value}
        step={step}
        className="h-9"
        onChange={(e) => {
          const v = parseFloat(e.target.value);
          if (Number.isFinite(v)) onChange(v);
        }}
      />
    </div>
  );
}

export interface BoundaryEditorProps {
  faces: Record<FaceKey, FaceBC>;
  pinGauge: boolean;
  onFaceChange: (face: FaceKey, patch: Partial<FaceBC>) => void;
  onPinChange: (v: boolean) => void;
}

export function BoundaryEditor({
  faces, pinGauge, onFaceChange, onPinChange,
}: BoundaryEditorProps) {
  const anyDirichlet = FACE_KEYS.some((f) => faces[f].mode === "dirichlet");
  return (
    <div className="rounded-md border border-border bg-muted/20 p-3 space-y-3">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <div className="text-[11px] uppercase tracking-wide text-muted-foreground">
            Boundary conditions per bbox face
          </div>
          <div className="text-[11px] text-muted-foreground">
            Dirichlet pins φ. Neumann prescribes v·n_out (m/s): positive = outflow, negative = inflow. Wall = ∂φ/∂n = 0.
          </div>
        </div>
        {!anyDirichlet && (
          <label className="flex items-center gap-2 text-xs">
            <input
              type="checkbox"
              checked={pinGauge}
              onChange={(e) => onPinChange(e.target.checked)}
            />
            Auto-pin gauge (vertex 0 → φ=0)
          </label>
        )}
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
        {FACE_KEYS.map((f) => {
          const bc = faces[f];
          return (
            <div key={f} className="rounded border border-border/60 bg-background/40 p-2 space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-mono text-sm text-foreground">face {f}</span>
                <Select
                  value={bc.mode}
                  onValueChange={(v) => onFaceChange(f, { mode: v as FaceMode })}
                >
                  <SelectTrigger className="h-8 w-[180px] text-xs"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="dirichlet">{MODE_LABEL.dirichlet}</SelectItem>
                    <SelectItem value="neumann">{MODE_LABEL.neumann}</SelectItem>
                    <SelectItem value="wall">{MODE_LABEL.wall}</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              {bc.mode === "dirichlet" && (
                <NumField
                  label="φ (m²/s)" value={bc.phi} step={0.1}
                  onChange={(v) => onFaceChange(f, { phi: v })}
                />
              )}
              {bc.mode === "neumann" && (
                <NumField
                  label="v·n_out (m/s) — inflow < 0" value={bc.vN} step={0.1}
                  onChange={(v) => onFaceChange(f, { vN: v })}
                />
              )}
              {bc.mode === "wall" && (
                <div className="text-[11px] text-muted-foreground italic">
                  No-penetration wall (insulated). Streamlines tangent to face.
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
