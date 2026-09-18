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

// ---- jobs / runs / params (M3) ----
export const jobStatusSchema = z.enum(["queued", "running", "succeeded", "failed", "cancelled", "orphaned"]);
export const jobKindSchema = z.enum([
  "merge",
  "append",
  "consolidate",
  "official_eval",
  "import_results",
  "export",
  "export_verify",
]);
export const jobProgressSchema = z.object({
  step: z.number().nullable(),
  total: z.number().nullable(),
  stage_index: z.number().nullable(),
  stage: z.string().nullable(),
  completed_steps: z.number(),
  detail: z.string(),
});
export const jobSchema = z.object({
  id: z.string(),
  kind: jobKindSchema,
  queue: z.enum(["gpu", "cpu"]),
  status: jobStatusSchema,
  region_id: z.string().nullable(),
  run_id: z.string().nullable(),
  argv: z.array(z.string()),
  cwd: z.string(),
  env: z.record(z.string(), z.string()),
  cpu_list: z.string().nullable(),
  log_path: z.string(),
  pid: z.number().nullable(),
  process_create_time: z.number().nullable(),
  created_at: z.string(),
  started_at: z.string().nullable(),
  finished_at: z.string().nullable(),
  returncode: z.number().nullable(),
  crash_kind: z.string().nullable(),
  error: z.string().nullable(),
  progress: jobProgressSchema,
});
export const stepRecordSchema = z.object({
  index: z.number(),
  session_id: z.string(),
  dir_name: z.string(),
  status: z.enum(["running", "done", "failed"]),
  id_offset: z.number().nullable(),
  odom_nodes: z.number().nullable(),
  covis_nodes: z.number().nullable(),
  components: z.number().nullable(),
  registry_edges: z.number().nullable(),
  pgo_error_initial: z.number().nullable(),
  pgo_error_final: z.number().nullable(),
  started_at: z.string().nullable(),
  finished_at: z.string().nullable(),
});
export const runParentSchema = z.object({ run_id: z.string(), step_index: z.number() });
export const runStatusSchema = z.enum(["queued", "running", "succeeded", "failed", "cancelled", "orphaned"]);
export const runSchema = z.object({
  id: z.string(),
  region_id: z.string(),
  name: z.string(),
  kind: z.enum(["merge", "append", "imported"]),
  parent: runParentSchema.nullable(),
  start_step: z.number(),
  session_ids: z.array(z.string()),
  params: z.record(z.string(), z.unknown()),
  meta: z.record(z.string(), z.unknown()),
  status: runStatusSchema,
  job_id: z.string().nullable(),
  git_commit: z.string().nullable(),
  num_steps_expected: z.number(),
  last_step_index: z.number().nullable(),
  final_dir: z.string().nullable(),
  final_error: z.string().nullable(),
  created_at: z.string(),
  finished_at: z.string().nullable(),
});
export const runDetailSchema = z.object({
  run: runSchema,
  steps: z.array(stepRecordSchema),
  job: jobSchema.nullable(),
});
export const paramSpecSchema = z.object({
  name: z.string(),
  type: z.enum(["int", "float", "bool", "choice", "str"]),
  default: z.unknown(),
  help: z.string(),
  group: z.string(),
  choices: z.array(z.string()).nullable(),
  advanced: z.boolean(),
});
export const logChunkSchema = z.object({ lines: z.array(z.string()), next: z.number(), total: z.number() });
