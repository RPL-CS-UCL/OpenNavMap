import type { z } from "zod";
import type {
  fileCheckSchema,
  fsEntrySchema,
  fsListingSchema,
  healthSchema,
  regionHeadSchema,
  regionSchema,
  sessionSchema,
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
