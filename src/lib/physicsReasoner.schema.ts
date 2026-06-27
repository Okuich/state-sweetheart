/**
 * Shared Zod schema for physics reasoner recommendations.
 * Lives outside src/server/ so client-safe modules (route handlers,
 * UI panels) can import it without tripping import-protection.
 */
import { z } from "zod";

export const RecommendationSchema = z.object({
  rationale: z.string(),
  confidence: z.number().min(0).max(1),
  parameters: z.array(z.object({
    name: z.string(),
    value: z.union([z.number(), z.string()]),
    unit: z.string().optional(),
    rationale: z.string(),
  })).min(1).max(12),
  warnings: z.array(z.string()).default([]),
  nextActions: z.array(z.string()).default([]),
});
export type Recommendation = z.infer<typeof RecommendationSchema>;
