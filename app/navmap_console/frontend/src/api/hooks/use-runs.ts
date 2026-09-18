import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { z } from "zod";
import { apiGet, apiSend } from "../client";
import { qk } from "../query-keys";
import { runDetailSchema, runSchema } from "../schemas";
import type { Run, RunParent } from "../types";

export interface RunCreateBody {
  name?: string;
  kind: "merge" | "append";
  parent?: RunParent | null;
  session_ids: string[];
  params: Record<string, unknown>;
  meta?: Record<string, unknown>;
}

export function useRuns(rid: string) {
  return useQuery({
    queryKey: qk.runs.list(rid),
    queryFn: () => apiGet(`/api/regions/${rid}/runs`, z.array(runSchema)),
    enabled: !!rid,
  });
}

export function useRunDetail(rid: string, runId: string) {
  return useQuery({
    queryKey: qk.runs.one(rid, runId),
    queryFn: () => apiGet(`/api/regions/${rid}/runs/${runId}`, runDetailSchema),
    enabled: !!rid && !!runId,
  });
}

export function useCreateRun(rid: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: RunCreateBody) => apiSend<Run>("POST", `/api/regions/${rid}/runs`, body, runSchema),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.runs.list(rid) });
      qc.invalidateQueries({ queryKey: qk.regions.one(rid) });
      qc.invalidateQueries({ queryKey: qk.jobs.all });
    },
  });
}

export function useCancelRun(rid: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (runId: string) =>
      apiSend<Run>("POST", `/api/regions/${rid}/runs/${runId}/cancel`, undefined, runSchema),
    onSuccess: (run) => qc.invalidateQueries({ queryKey: qk.runs.one(rid, run.id) }),
  });
}
