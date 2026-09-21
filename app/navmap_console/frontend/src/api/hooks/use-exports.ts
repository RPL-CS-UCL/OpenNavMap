import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiGet, apiSend } from "../client";
import { qk } from "../query-keys";
import { exportsSchema, jobSchema } from "../schemas";
import type { ExportKind, Job } from "../types";

const base = (rid: string, runId: string) => `/api/regions/${rid}/runs/${runId}/exports`;

export function useExports(rid: string, runId: string) {
  return useQuery({
    queryKey: qk.exports.list(rid, runId),
    queryFn: () => apiGet(base(rid, runId), exportsSchema),
    enabled: !!rid && !!runId,
    // bundles pack in a cpu job; poll while one is in flight so size/status land without a reload
    refetchInterval: (query) =>
      query.state.data?.items.some((e) => e.status === "queued" || e.status === "running") ? 3000 : false,
  });
}

export function useCreateExport(rid: string, runId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { kind: ExportKind; steps?: number[] }) =>
      apiSend<Job>("POST", base(rid, runId), body, jobSchema),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: qk.exports.list(rid, runId) });
      void qc.invalidateQueries({ queryKey: qk.jobs.all });
    },
  });
}

export function useDeleteExport(rid: string, runId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (name: string) => apiSend<void>("DELETE", `${base(rid, runId)}/${name}`),
    onSuccess: () => void qc.invalidateQueries({ queryKey: qk.exports.list(rid, runId) }),
  });
}

export const exportDownloadUrl = (rid: string, runId: string, name: string) => `${base(rid, runId)}/${name}/download`;
