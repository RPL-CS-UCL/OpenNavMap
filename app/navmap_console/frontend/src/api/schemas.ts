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
  "per_step_eval",
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
export const loopStatsSchema = z.object({
  total: z.number(), new: z.number(), hist: z.number(), accepted: z.number(),
  rejected_new: z.number(), overturned_hist: z.number(),
});
export const stepSummarySchema = z.object({
  index: z.number(),
  dir_name: z.string(),
  session_id: z.string(),
  status: z.string(),
  id_offset: z.number(),
  num_nodes: z.number(),
  num_covis_nodes: z.number(),
  num_new: z.number(),
  num_culled: z.number(),
  component_sizes: z.array(z.number()),
  edge_counts: z.record(z.string(), z.number()),
  history: z.record(z.string(), z.number()),
  precision: z.array(z.number()),
  recall: z.array(z.number()),
  loops: loopStatsSchema,
  pgo_error_initial: z.number().nullable(),
  pgo_error_final: z.number().nullable(),
  max_displacement: z.number(),
  num_moved: z.number(),
  duration_s: z.number().nullable(),
  has_dmatrix: z.boolean(),
  has_pre_pgo: z.boolean(),
  // per-step ATE from the traj_evaluation toolchain; null until that step's eval job lands
  ate_trans_rmse: z.number().nullable(),
  ate_rot_rmse: z.number().nullable(),
  ate_frames: z.number().nullable(),
  ate_reason: z.string().nullable(),
});
export const summariesSchema = z.object({ steps: z.array(stepSummarySchema) });
/** One evaluation report (only "final" exists today); numbers come from eval.json written by the eval job. */
export const evaluationSchema = z.object({
  eid: z.string(),
  status: z.string(),
  ate_trans: z.number().nullable().optional(),
  ate_rot: z.number().nullable().optional(),
  frames: z.number().nullable().optional(),
  created_at: z.string().nullable().optional(),
  error: z.string().nullable().optional(),
  job: jobSchema.nullable().optional(),
  report_files: z.array(z.string()).optional(),
});
export const evaluationsSchema = z.object({ items: z.array(evaluationSchema), evaluating: z.boolean() });
export const exportKindSchema = z.enum(["map", "report", "preds"]);
/** One download bundle under exports/; metadata json written by export_job.py (size/sha256 once packed). */
export const exportItemSchema = z.object({
  name: z.string(),
  kind: exportKindSchema,
  status: z.enum(["queued", "running", "done", "failed", "cancelled"]),
  created_at: z.string().nullable().optional(),
  job_id: z.string().nullable().optional(),
  steps: z.array(z.number()).nullable().optional(),
  entries: z.number().nullable().optional(),
  size: z.number().nullable().optional(),
  sha256: z.string().nullable().optional(),
  verified: z.boolean().nullable().optional(),
  error: z.string().nullable().optional(),
});
export const exportsSchema = z.object({ items: z.array(exportItemSchema) });
export const dmatrixCandidateSchema = z.object({ db: z.number(), query: z.number(), stage: z.string(), gv_inliers: z.number() });
export const dmatrixFactorSchema = z.object({
  db: z.number(), query: z.number(), weight: z.number(), conf: z.number(), accepted: z.boolean(), origin: z.string(),
});
export const dmatrixSchema = z.object({
  rows: z.string(),
  cols: z.string(),
  row_node_ids: z.array(z.number()),
  col_node_ids: z.array(z.number()),
  vmin: z.number(),
  vmax: z.number(),
  candidates: z.array(dmatrixCandidateSchema),
  factors: z.array(dmatrixFactorSchema),
});
export const cullRowSchema = z.object({
  node_id: z.number(),
  kind: z.string(),
  other: z.number().nullable(),
  prob: z.number().nullable(),
  method: z.string(),
  detail: z.string(),
  image_url: z.string(),
  other_image_url: z.string().nullable(),
  vis_url: z.string().nullable(),
});
export const cullingSchema = z.object({ culled: z.array(cullRowSchema), kept: z.array(cullRowSchema) });
const latLon = z.tuple([z.number(), z.number()]);
/** GPS-aligned trajectory of one step (readers/geo.py): traj has one [lat, lon] per frame, gps the raw fixes. */
export const geoSchema = z.object({
  origin: latLon.nullable(),
  n_frames: z.number(),
  n_gps: z.number(),
  traj: z.array(latLon),
  gps: z.array(latLon),
  rmse_m: z.number().nullable(),
  reason: z.string().nullable(),
});
export const nodeLoopSchema = z.object({
  other: z.number(), weight: z.number(), conf: z.number(), accepted: z.boolean(), origin: z.string(),
});
export const nodeDetailSchema = z.object({
  node_id: z.number(),
  step: z.number(),
  session_id: z.string(),
  frame: z.string(),
  timestamp: z.number().nullable(),
  pos: z.array(z.number()),
  quat: z.array(z.number()),
  pos_pre: z.array(z.number()),
  gt: z.array(z.number()).nullable(),
  degree: z.object({ odom: z.number(), covis: z.number(), trav: z.number() }),
  flags: z.number(),
  image_url: z.string(),
  cull: z.object({ other: z.number().nullable(), prob: z.number().nullable(), method: z.string(), detail: z.string() }).nullable(),
  loops: z.array(nodeLoopSchema),
});
export const runEventSchema = z.object({
  demo_step: z.number().nullable().optional(),
  merge_step: z.number(),
  stage: z.string().nullable().optional(),
  event_type: z.string(),
  submap_id: z.union([z.number(), z.string()]).nullable().optional(),
  keyframe_id: z.union([z.number(), z.string()]).nullable().optional(),
  payload: z.record(z.string(), z.unknown()),
  artifacts: z.record(z.string(), z.unknown()).optional(),
});
