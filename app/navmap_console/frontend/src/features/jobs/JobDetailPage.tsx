import { Fragment, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { toast } from "sonner";
import { errorDetail } from "@/api/client";
import { useCancelJob, useJob } from "@/api/hooks/use-jobs";
import { ConfirmDialog } from "@/components/common/ConfirmDialog";
import { PageHeader } from "@/components/common/PageHeader";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { t } from "@/i18n";
import { LogConsole } from "./LogConsole";
import { JobStatusBadge, crashHint, progressText } from "./job-status";
import { durationOf } from "./JobsPage";
import { useJobLog } from "./use-job-log";

export function JobDetailPage() {
  const { jid = "" } = useParams();
  const job = useJob(jid);
  const cancel = useCancelJob();
  const [confirm, setConfirm] = useState(false);
  const { lines } = useJobLog(jid || null);

  if (job.isPending) return <Skeleton className="h-40 w-full" />;
  if (job.error) return <p className="text-sm text-destructive">{t("common.error", { detail: job.error.message })}</p>;
  const j = job.data;
  const active = j.status === "queued" || j.status === "running";
  const hint = j.status === "failed" ? crashHint(j.crash_kind) : null;
  const stage = j.progress.stage ? ` · ${j.progress.stage}` : "";

  return (
    <div className="flex h-full min-h-0 flex-col">
      <PageHeader
        backTo="/jobs"
        title={j.id}
        description={`${j.kind} · ${progressText(j.progress)}${stage} · ${durationOf(j.started_at, j.finished_at)}`}
        actions={
          <>
            <JobStatusBadge status={j.status} />
            {j.run_id && j.region_id && (
              <Button asChild size="sm" variant="outline">
                <Link to={`/regions/${j.region_id}/runs/${j.run_id}`}>{t("jobs.detail.openRun")}</Link>
              </Button>
            )}
            {active && (
              <Button size="sm" variant="destructive" onClick={() => setConfirm(true)}>
                {t("jobs.cancel")}
              </Button>
            )}
          </>
        }
      />
      {hint && (
        <Alert variant="destructive" className="mb-3">
          <AlertTitle>{j.crash_kind}</AlertTitle>
          <AlertDescription>{hint}</AlertDescription>
        </Alert>
      )}
      <details className="mb-3 text-xs">
        <summary className="cursor-pointer text-muted-foreground">{t("jobs.detail.command")}</summary>
        <pre className="mt-1 overflow-auto rounded-sm bg-muted p-2 font-mono text-[11px]">{j.argv.join(" \\\n  ")}</pre>
        <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
          <dt className="text-muted-foreground">{t("jobs.detail.cpus")}</dt>
          <dd className="font-mono">{j.cpu_list ?? "-"}</dd>
          <dt className="text-muted-foreground">{t("jobs.detail.pid")}</dt>
          <dd className="font-mono">{j.pid ?? "-"}</dd>
          <dt className="text-muted-foreground">{t("jobs.detail.returncode")}</dt>
          <dd className="font-mono">{j.returncode ?? "-"}</dd>
          {Object.entries(j.env).map(([k, v]) => (
            <Fragment key={k}>
              <dt className="font-mono text-muted-foreground">{k}</dt>
              <dd className="break-all font-mono">{v}</dd>
            </Fragment>
          ))}
        </dl>
      </details>
      <LogConsole lines={lines} title={j.id} className="flex-1" />
      <ConfirmDialog
        open={confirm}
        onOpenChange={setConfirm}
        title={t("jobs.cancel")}
        description={t("jobs.cancel.confirm")}
        destructive
        pending={cancel.isPending}
        onConfirm={() =>
          cancel.mutate(j.id, {
            onSuccess: () => {
              toast.success(t("jobs.cancelled", { id: j.id }));
              setConfirm(false);
            },
            onError: (err) => toast.error(errorDetail(err)),
          })
        }
      />
    </div>
  );
}
