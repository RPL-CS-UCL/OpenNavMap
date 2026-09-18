import { z } from "zod";

export const vprConfigSchema = z.object({ method: z.string(), backbone: z.string(), dim: z.number() });
export const regionHeadSchema = z.object({
  run_id: z.string(),
  step_index: z.number(),
  session_ids: z.array(z.string()),
  lineage: z.array(z.string()),
});
export const regionSchema = z.object({
  id: z.string(),
  name: z.string(),
  description: z.string(),
  vpr: vprConfigSchema,
  image_size: z.array(z.number()),
  created_at: z.string(),
  head: regionHeadSchema.nullable(),
  session_count: z.number().optional(),
  run_count: z.number().optional(),
});
export const fileCheckSchema = z.object({
  name: z.string(),
  required: z.boolean(),
  status: z.enum(["ok", "missing", "incomplete", "invalid"]),
  lines: z.number().nullable(),
  detail: z.string(),
});
export const validationReportSchema = z.object({
  ok: z.boolean(),
  num_frames: z.number(),
  has_gt: z.boolean(),
  has_gps: z.boolean(),
  has_iqa: z.boolean(),
  descriptor_dim: z.number().nullable(),
  files: z.array(fileCheckSchema),
  warnings: z.array(z.string()),
  errors: z.array(z.string()),
  checked_at: z.string(),
});
export const sessionSchema = z.object({
  id: z.string(),
  region_id: z.string(),
  name: z.string(),
  kind: z.enum(["submap", "raw"]),
  source: z.enum(["upload", "path", "imported"]),
  path: z.string(),
  num_frames: z.number(),
  has_gt: z.boolean(),
  has_gps: z.boolean(),
  has_iqa: z.boolean(),
  validation: validationReportSchema.nullable(),
  created_at: z.string(),
});
export const fsEntrySchema = z.object({
  name: z.string(),
  path: z.string(),
  is_dir: z.boolean(),
  looks_like_session: z.boolean(),
  size: z.number().nullable(),
});
export const fsListingSchema = z.object({
  path: z.string(),
  parent: z.string().nullable(),
  entries: z.array(fsEntrySchema),
});
export const fsRootsSchema = z.object({ roots: z.array(z.string()) });
export const healthSchema = z.object({
  version: z.string(),
  data_root: z.string(),
  repo_root: z.string(),
  queues: z.object({ gpu: z.number(), cpu: z.number() }),
  gpu: z.object({ name: z.string(), memory_used_mib: z.number(), memory_total_mib: z.number() }).nullable(),
  disk: z.object({ total: z.number(), free: z.number() }),
  cpu_list: z.string().nullable(),
});
