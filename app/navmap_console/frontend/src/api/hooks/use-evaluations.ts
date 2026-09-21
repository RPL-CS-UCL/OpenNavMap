import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiGet, apiSend } from "../client";
import { qk } from "../query-keys";
import { evaluationsSchema, jobSchema } from "../schemas";
import type { Job } from "../types";

const base = (rid: string, runId: string) => `/api/regions/${rid}/runs/${runId}`;

export function useEvaluations(rid: string, runId: string) {
  return useQuery({
    queryKey: qk.evaluations.list(rid, runId),
    queryFn: () => apiGet(`${base(rid, runId)}/evaluations`, evaluationsSchema),
    enabled: !!rid && !!runId,
    // the run.evaluated socket event is the usual trigger; poll while a report is in flight as a fallback
    refetchInterval: (query) => (query.state.data?.evaluating ? 5000 : false),
  });
}

/** Queue the final evaluation again (e.g. after the toolchain was fixed); 404 when the run has no GT. */
export function useRerunEvaluation(rid: string, runId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => apiSend<Job>("POST", `${base(rid, runId)}/evaluate`, undefined, jobSchema),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: qk.evaluations.list(rid, runId) });
      void qc.invalidateQueries({ queryKey: qk.jobs.all });
    },
  });
}

export const evaluationFileUrl = (rid: string, runId: string, eid: string, name: string) =>
  `${base(rid, runId)}/evaluations/${eid}/files/${name}`;
