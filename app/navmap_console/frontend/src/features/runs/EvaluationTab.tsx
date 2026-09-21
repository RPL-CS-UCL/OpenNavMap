import { X } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import { errorDetail } from "@/api/client";
import { evaluationFileUrl, useEvaluations, useRerunEvaluation } from "@/api/hooks/use-evaluations";
import { EmptyState } from "@/components/common/EmptyState";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { JobStatusBadge } from "@/features/jobs/job-status";
import { t } from "@/i18n";

const PREVIEWABLE = /\.(png|pdf|jpe?g|svg)$/i;
const IS_PDF = /\.pdf$/i;

/** Final-map ATE from the traj_evaluation toolchain: numbers, report files and a re-run button. */
export function EvaluationTab({ rid, runId }: { rid: string; runId: string }) {
  const evals = useEvaluations(rid, runId);
  const rerun = useRerunEvaluation(rid, runId);
  const [previewed, setPreviewed] = useState<string | null>(null); // report file shown inline
  const rerunButton = (
    <Button
      size="sm"
      variant="outline"
      disabled={rerun.isPending}
      onClick={() => rerun.mutate(undefined, {
        onSuccess: () => toast.success(t("run.evaluation.queued")),
        onError: (err) => toast.error(errorDetail(err)),
      })}
    >
      {t("run.evaluation.rerun")}
    </Button>
  );
  if (evals.isPending) return <Skeleton className="m-3 h-16" />;
  if (evals.isError) return <EmptyState title={t("common.error", { detail: errorDetail(evals.error) })} action={rerunButton} />;
  const item = evals.data.items[0];
  const hasNumbers = item !== undefined && item.ate_trans !== undefined && item.ate_trans !== null;
  if (!item) {
    return <EmptyState title={t("run.evaluation.none")} body={t("run.evaluation.noGt")} action={rerunButton} />;
  }
  return (
    <div className="flex min-h-0 flex-col gap-3 p-3 text-sm">
      <div className="flex items-center gap-2">
        {item.job && <JobStatusBadge status={item.job.status} />}
        {item.created_at && <span className="text-xs text-muted-foreground">{item.created_at}</span>}
        {rerunButton}
      </div>
      {item.status === "failed" && item.error && (
        <p className="rounded-sm border border-destructive/40 p-2 font-mono text-xs text-destructive">{item.error}</p>
      )}
      {hasNumbers ? (
        <dl className="grid grid-cols-3 gap-2 font-mono tabular-nums">
          <div>
            <dt className="text-xs text-muted-foreground">{t("run.evaluation.ateTrans")}</dt>
            <dd data-testid="ate-trans">{item.ate_trans!.toFixed(3)} m</dd>
          </div>
          <div>
            <dt className="text-xs text-muted-foreground">{t("run.evaluation.ateRot")}</dt>
            <dd data-testid="ate-rot">{item.ate_rot?.toFixed(2) ?? "–"} °</dd>
          </div>
          <div>
            <dt className="text-xs text-muted-foreground">{t("run.evaluation.frames")}</dt>
            <dd data-testid="ate-frames">{item.frames ?? "–"}</dd>
          </div>
        </dl>
      ) : (
        item.status !== "failed" && <p className="text-xs text-muted-foreground">{t("run.evaluation.pending")}</p>
      )}
      {previewed && (
        <div className="flex flex-col gap-1" data-testid="report-preview">
          <div className="flex items-center justify-between gap-2 font-mono text-xs">
            <span className="truncate">{previewed}</span>
            <Button size="sm" variant="ghost" aria-label={t("run.evaluation.closePreview")} onClick={() => setPreviewed(null)}>
              <X className="h-4 w-4" />
            </Button>
          </div>
          {IS_PDF.test(previewed) ? (
            <iframe src={evaluationFileUrl(rid, runId, item.eid, previewed)} title={previewed} className="h-[70vh] w-full rounded-sm border" />
          ) : (
            <img src={evaluationFileUrl(rid, runId, item.eid, previewed)} alt={previewed} className="max-h-[70vh] w-fit rounded-sm border" />
          )}
        </div>
      )}
      {item.report_files && item.report_files.length > 0 && (
        <ul className="flex flex-col gap-1">
          {item.report_files.map((f) => (
            <li key={f} className="flex items-center justify-between gap-2 rounded-sm border px-2 py-1 font-mono text-xs">
              <span className="truncate">{f}</span>
              <div className="flex shrink-0 gap-1">
                {PREVIEWABLE.test(f) && (
                  <Button size="sm" variant={previewed === f ? "secondary" : "ghost"} onClick={() => setPreviewed(previewed === f ? null : f)}>
                    {t("run.evaluation.preview")}
                  </Button>
                )}
                <Button size="sm" variant="ghost" asChild>
                  <a href={evaluationFileUrl(rid, runId, item.eid, f)} download>{t("run.evaluation.download")}</a>
                </Button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
