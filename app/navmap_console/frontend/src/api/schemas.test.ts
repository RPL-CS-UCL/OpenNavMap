import { describe, expect, it } from "vitest";
import { jobSchema, paramSpecSchema, runDetailSchema, runSchema, stepSummarySchema } from "./schemas";
import { job, paramSpecs, run, steps, summaries } from "@/test/fixtures";

describe("schemas", () => {
  it("parse the job/run/step fixtures", () => {
    expect(jobSchema.parse(job).progress.total).toBe(2);
    expect(runSchema.parse(run).kind).toBe("merge");
    expect(runDetailSchema.parse({ run, steps, job }).steps).toHaveLength(2);
    expect(paramSpecs.map((p) => paramSpecSchema.parse(p).type)).toContain("choice");
  });

  it("parses step summaries", () => {
    expect(stepSummarySchema.array().parse(summaries)).toHaveLength(2);
  });
});
