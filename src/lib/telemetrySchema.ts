import { z } from "zod";

export const TelemetrySampleSchema = z.object({
  t: z.number().finite(),
  energy_drift_pct: z.number().finite().optional(),
  constraint_l2:    z.number().finite().min(0).optional(),
  divergence_risk:  z.number().finite().min(0).max(1).optional(),
  velocity_max:     z.number().finite().min(0).optional(),
  nan_count:        z.number().int().min(0).optional(),
  source:           z.string().min(1).max(64).optional(),
});
export type TelemetrySample = z.infer<typeof TelemetrySampleSchema>;

export const IngestSchema = z.union([
  TelemetrySampleSchema,
  z.object({ samples: z.array(TelemetrySampleSchema).min(1).max(256) }),
]);
