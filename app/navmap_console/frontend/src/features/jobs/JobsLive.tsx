import { useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { qk } from "@/api/query-keys";
import type { Job, JobProgress } from "@/api/types";
import { t } from "@/i18n";
import { useTopic } from "@/ws/use-socket";

/** Turns `jobs` topic events into query invalidations and toasts; mounted once in AppShell. */
export function JobsLive() {
  const qc = useQueryClient();
  useTopic("jobs", (m) => {
    if (m.type === "job.state") {
      const job = m.data as Job;
      qc.invalidateQueries({ queryKey: qk.jobs.all });
      qc.setQueryData(qk.jobs.one(job.id), job);
      if (job.region_id && job.run_id) qc.invalidateQueries({ queryKey: qk.runs.one(job.region_id, job.run_id) });
      if (job.region_id) qc.invalidateQueries({ queryKey: qk.regions.one(job.region_id) });
      if (job.status === "succeeded") toast.success(t("jobs.finished", { kind: job.kind, id: job.id }));
      if (job.status === "failed" || job.status === "orphaned") {
        toast.error(t("jobs.failed", { kind: job.kind, id: job.id, status: job.status }));
      }
    } else if (m.type === "job.progress") {
      const { id, progress } = m.data as { id: string; progress: JobProgress };
      qc.setQueryData<Job>(qk.jobs.one(id), (old) => (old ? { ...old, progress } : old));
      qc.setQueriesData<Job[]>({ queryKey: ["jobs", "list"] }, (old) =>
        old?.map((j) => (j.id === id ? { ...j, progress } : j)),
      );
    }
  });
  return null;
}
