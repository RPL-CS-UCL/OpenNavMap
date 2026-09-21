import type { z } from "zod";
import type {
  cullRowSchema,
  cullingSchema,
  dmatrixCandidateSchema,
  dmatrixFactorSchema,
  dmatrixSchema,
  evaluationSchema,
  evaluationsSchema,
  exportItemSchema,
  exportKindSchema,
  fileCheckSchema,
  geoSchema,
  fsEntrySchema,
  fsListingSchema,
  healthSchema,
  jobKindSchema,
  jobProgressSchema,
  jobSchema,
  jobStatusSchema,
  logChunkSchema,
  loopStatsSchema,
  nodeDetailSchema,
  nodeLoopSchema,
  paramSpecSchema,
  regionHeadSchema,
  regionSchema,
  runDetailSchema,
  runEventSchema,
  runParentSchema,
  runSchema,
  runStatusSchema,
  sessionSchema,
  stepRecordSchema,
  stepSummarySchema,
  validationReportSchema,
  vprConfigSchema,
} from "./schemas";

export type VprConfig = z.infer<typeof vprConfigSchema>;
export type RegionHead = z.infer<typeof regionHeadSchema>;
export type Region = z.infer<typeof regionSchema>;
export type FileCheck = z.infer<typeof fileCheckSchema>;
export type ValidationReport = z.infer<typeof validationReportSchema>;
export type Session = z.infer<typeof sessionSchema>;
export type FsEntry = z.infer<typeof fsEntrySchema>;
export type FsListing = z.infer<typeof fsListingSchema>;
export type Health = z.infer<typeof healthSchema>;
export type JobStatus = z.infer<typeof jobStatusSchema>;
export type JobKind = z.infer<typeof jobKindSchema>;
export type JobProgress = z.infer<typeof jobProgressSchema>;
export type Job = z.infer<typeof jobSchema>;
export type StepRecord = z.infer<typeof stepRecordSchema>;
export type RunParent = z.infer<typeof runParentSchema>;
export type RunStatus = z.infer<typeof runStatusSchema>;
export type Run = z.infer<typeof runSchema>;
export type RunDetail = z.infer<typeof runDetailSchema>;
export type ParamSpec = z.infer<typeof paramSpecSchema>;
export type LogChunk = z.infer<typeof logChunkSchema>;

export type LoopStats = z.infer<typeof loopStatsSchema>;
export type StepSummary = z.infer<typeof stepSummarySchema>;
export type DMatrixCandidate = z.infer<typeof dmatrixCandidateSchema>;
export type DMatrixFactor = z.infer<typeof dmatrixFactorSchema>;
export type DMatrix = z.infer<typeof dmatrixSchema>;
export type CullRow = z.infer<typeof cullRowSchema>;
export type Culling = z.infer<typeof cullingSchema>;
export type NodeLoop = z.infer<typeof nodeLoopSchema>;
export type NodeDetail = z.infer<typeof nodeDetailSchema>;
export type RunEvent = z.infer<typeof runEventSchema>;
export type Evaluation = z.infer<typeof evaluationSchema>;
export type Evaluations = z.infer<typeof evaluationsSchema>;
export type ExportKind = z.infer<typeof exportKindSchema>;
export type ExportItem = z.infer<typeof exportItemSchema>;
export type Geo = z.infer<typeof geoSchema>;
