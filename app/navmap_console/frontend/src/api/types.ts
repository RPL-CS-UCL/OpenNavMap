import type { z } from "zod";
import type {
  fileCheckSchema,
  fsEntrySchema,
  fsListingSchema,
  healthSchema,
  jobKindSchema,
  jobProgressSchema,
  jobSchema,
  jobStatusSchema,
  logChunkSchema,
  paramSpecSchema,
  regionHeadSchema,
  regionSchema,
  runDetailSchema,
  runParentSchema,
  runSchema,
  runStatusSchema,
  sessionSchema,
  stepRecordSchema,
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
