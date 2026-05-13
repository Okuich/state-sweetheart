/**
 * Server function: regenerate a mesh from input parameters and return an
 * exported file (VTK / OBJ / JSON). Deterministic — same input yields the
 * same output, so the client can re-request without sending the full mesh
 * payload over the wire.
 */

import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";
import { generateMesh } from "@/lib/meshing";
import { exportMesh, type ExportFormat } from "@/lib/meshing/export";

const Vec3 = z.tuple([z.number(), z.number(), z.number()]);

const SeedSchema = z.object({
  kind: z.enum(["hole", "fillet", "sharp", "hotspot", "overhang", "thin_wall", "contact"]),
  center: Vec3,
  radius: z.number().positive().max(1e6),
  weight: z.number().min(0).max(10),
});

const InputSchema = z.object({
  format: z.enum(["vtk", "obj", "json"]),
  bbox: z.object({ min: Vec3, max: Vec3 }),
  seeds: z.array(SeedSchema).max(64),
  octree: z
    .object({
      minDepth: z.number().int().min(0).max(8).optional(),
      maxDepth: z.number().int().min(0).max(8).optional(),
      refineThreshold: z.number().min(0).max(1).optional(),
      maxLeaves: z.number().int().min(8).max(200_000).optional(),
    })
    .optional(),
  partitionCount: z.number().int().min(1).max(64).optional(),
});

export const exportMeshFn = createServerFn({ method: "POST" })
  .inputValidator((input: unknown) => InputSchema.parse(input))
  .handler(async ({ data }) => {
    try {
      const result = generateMesh({
        bbox: data.bbox,
        seeds: data.seeds,
        octree: data.octree,
        partitionCount: data.partitionCount,
      });
      const file = exportMesh(result, data.format as ExportFormat);
      return {
        ok: true as const,
        filename: file.filename,
        mimeType: file.mimeType,
        content: file.content,
        bytes: file.content.length,
        summary: {
          leaves: result.summary.octree.leafCount,
          tets: result.summary.tets.count,
          vertices: result.summary.tets.vertexCount,
          partitions: result.summary.partition.partitionCount,
        },
      };
    } catch (err) {
      console.error("exportMeshFn failed:", err);
      return {
        ok: false as const,
        error: err instanceof Error ? err.message : "Mesh export failed",
      };
    }
  });
