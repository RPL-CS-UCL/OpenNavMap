import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { toast } from "sonner";
import { errorDetail } from "@/api/client";
import { usePromoteHead } from "@/api/hooks/use-regions";
import { useCancelRun, useRunDetail } from "@/api/hooks/use-runs";
import { useSessions } from "@/api/hooks/use-sessions";
import { qk } from "@/api/query-keys";
import { ConfirmDialog } from "@/components/common/ConfirmDialog";
import { EmptyState } from "@/components/common/EmptyState";
import { PageHeader } from "@/components/common/PageHeader";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { JobStatusBadge, crashHint } from "@/features/jobs/job-status";
import { LogConsole } from "@/features/jobs/LogConsole";
import { useJobLog } from "@/features/jobs/use-job-log";
import { t } from "@/i18n";
import { useTopic } from "@/ws/use-socket";
import { EvaluationTab } from "./EvaluationTab";
import { ExportTab } from "./ExportTab";
import { StepsTable } from "./StepsTable";
import { RunViewer } from "./viewer/RunViewer";

export function RunDetailPage() {
  const { rid = "", runId = "" } = useParams();
  const [sp] = useSearchParams();
  const qc = useQueryClient();
  const detail = useRunDetail(rid, runId);
  const sessions = useSessions(rid);
  const cancel = useCancelRun(rid);
  const promote = usePromoteHead(rid);
  const [confirm, setConfirm] = useState(false);
  const jobId = detail.data?.job?.id ?? null;
  const { lines } = useJobLog(jobId);
  // Any run event (step completed, state change) refetches the detail; the payload itself is not needed.
  useTopic(runId ? `run:${runId}` : null, () => qc.invalidateQueries({ queryKey: qk.runs.one(rid, runId) }));

  if (detail.isPending) return <Skeleton className="h-40 w-full" />;
  if (detail.error) return <p className="text-sm text-destructive">{t("common.error", { detail: detail.error.message })}</p>;
  const { run, steps, job } = detail.data;
  const active = run.status === "queued" || run.status === "running";
  const done = steps.filter((s) => s.status === "done").length;
  const hint = job?.status === "failed" ? crashHint(job.crash_kind) : null;
  const names = Object.fromEntries((sessions.data ?? []).map((s) => [s.id, s.name]));
  const last = run.last_step_index;
  const parentText = run.parent ? ` · from ${run.parent.run_id} step ${run.parent.step_index}` : "";
  const commitText = run.git_commit ? ` · @${run.git_commit.slice(0, 7)}` : "";

  return (
    <div className="flex h-full min-h-0 flex-col">
      <PageHeader
        backTo={`/regions/${rid}`}
        title={run.name || run.id}
        description={`${run.kind} · ${run.id}${parentText}${commitText}`}
        actions={
          <>
            <JobStatusBadge status={run.status} />
            {job && (
              <Button asChild size="sm" variant="outline">
                <Link to={`/jobs/${job.id}`}>{t("run.openJob")}</Link>
              </Button>
            )}
            {run.status === "succeeded" && last !== null && (
              <Button
                size="sm"
                variant="outline"
                disabled={promote.isPending}
                onClick={() =>
                  promote.mutate(
                    { run_id: run.id, step_index: last },
                    {
                      onSuccess: () => toast.success(t("run.promoted", { run: run.id, step: last })),
                      onError: (err) => toast.error(errorDetail(err)),
                    },
                  )
                }
              >
                {t("run.promote")}
              </Button>
            )}
            {active && (
              <Button size="sm" variant="destructive" onClick={() => setConfirm(true)}>
                {t("run.cancel")}
              </Button>
            )}
          </>
        }
      />
      <div className="mb-3 flex items-center gap-3 text-xs">
        <Progress value={run.num_steps_expected ? (done / run.num_steps_expected) * 100 : 0} className="h-1.5 w-64" />
        <span className="font-mono tabular-nums">{t("run.progress", { done, total: run.num_steps_expected })}</span>
        {job?.progress.stage && active && <span className="text-muted-foreground">{job.progress.stage}</span>}
      </div>
      {hint && (
        <Alert variant="destructive" className="mb-3">
          <AlertTitle>{job?.crash_kind}</AlertTitle>
          <AlertDescription>{hint}</AlertDescription>
        </Alert>
      )}
      {run.final_error && (
        <Alert variant="destructive" className="mb-3">
          <AlertDescription>{t("run.final.error", { detail: run.final_error })}</AlertDescription>
        </Alert>
      )}
      <Tabs defaultValue={sp.has("step") ? "viz" : "steps"} className="flex min-h-0 flex-1 flex-col">
        <TabsList className="w-fit">
          <TabsTrigger value="steps">{t("run.steps.title")}</TabsTrigger>
          <TabsTrigger value="log">{t("run.log")}</TabsTrigger>
          <TabsTrigger value="viz">3D</TabsTrigger>
          <TabsTrigger value="evaluation">{t("run.evaluation.title")}</TabsTrigger>
          <TabsTrigger value="export">{t("run.export.title")}</TabsTrigger>
        </TabsList>
        <TabsContent value="steps" className="min-h-0 overflow-auto">
          <StepsTable steps={steps} expected={run.num_steps_expected} startStep={run.start_step} sessionNames={names} />
        </TabsContent>
        <TabsContent value="log" className="min-h-0 flex-1">
          {job ? (
            <LogConsole lines={lines} title={job.id} className="h-[60vh]" />
          ) : (
            <EmptyState title={t("run.log.none")} body={t("run.log.imported")} />
          )}
        </TabsContent>
        <TabsContent value="viz" className="min-h-0 flex-1">
          <RunViewer rid={rid} runId={runId} logLines={lines} hasJob={!!job} />
        </TabsContent>
        <TabsContent value="evaluation" className="min-h-0 flex-1 overflow-auto">
          <EvaluationTab rid={rid} runId={runId} />
        </TabsContent>
        <TabsContent value="export" className="min-h-0 flex-1 overflow-auto">
          <ExportTab rid={rid} runId={runId} />
        </TabsContent>
      </Tabs>
      <ConfirmDialog
        open={confirm}
        onOpenChange={setConfirm}
        title={t("run.cancel")}
        description={t("run.cancel.confirm")}
        destructive
        pending={cancel.isPending}
        onConfirm={() =>
          cancel.mutate(run.id, {
            onSuccess: () => {
              toast.success(t("run.cancelled"));
              setConfirm(false);
            },
            onError: (err) => toast.error(errorDetail(err)),
          })
        }
      />
    </div>
  );
}
