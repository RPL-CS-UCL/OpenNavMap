import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { z } from "zod";
import { apiGet, apiSend } from "../client";
import { qk } from "../query-keys";
import { jobSchema, logChunkSchema } from "../schemas";
import type { Job, LogChunk } from "../types";

export function useJobs(status = "") {
  const qs = status ? `?status=${encodeURIComponent(status)}` : "";
  return useQuery({ queryKey: qk.jobs.list(status), queryFn: () => apiGet(`/api/jobs${qs}`, z.array(jobSchema)) });
}

export function useJob(jid: string) {
  return useQuery({ queryKey: qk.jobs.one(jid), queryFn: () => apiGet(`/api/jobs/${jid}`, jobSchema), enabled: !!jid });
}

export function fetchJobLog(jid: string, after: number, limit = 5000): Promise<LogChunk> {
  return apiGet(`/api/jobs/${jid}/log?after=${after}&limit=${limit}`, logChunkSchema);
}

export function useCancelJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jid: string) => apiSend<Job>("POST", `/api/jobs/${jid}/cancel`, undefined, jobSchema),
    onSuccess: (job) => {
      qc.invalidateQueries({ queryKey: qk.jobs.all });
      qc.setQueryData(qk.jobs.one(job.id), job);
    },
  });
}
