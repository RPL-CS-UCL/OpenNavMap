import { useState } from "react";
import { toast } from "sonner";
import { errorDetail } from "@/api/client";
import { exportDownloadUrl, useCreateExport, useDeleteExport, useExports } from "@/api/hooks/use-exports";
import type { ExportItem, ExportKind } from "@/api/types";
import { ConfirmDialog } from "@/components/common/ConfirmDialog";
import { EmptyState } from "@/components/common/EmptyState";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { JobStatusBadge } from "@/features/jobs/job-status";
import { t } from "@/i18n";
import { formatBytes } from "@/lib/format";

const KINDS: ExportKind[] = ["map", "report", "preds"];

/** Download bundles (map / report / preds): create, list, download and delete; the backend keeps the newest three. */
export function ExportTab({ rid, runId }: { rid: string; runId: string }) {
  const exports = useExports(rid, runId);
  const create = useCreateExport(rid, runId);
  const remove = useDeleteExport(rid, runId);
  const [pendingDelete, setPendingDelete] = useState<ExportItem | null>(null);

  const createButtons = (
    <div className="flex flex-wrap gap-2">
      {KINDS.map((kind) => (
        <Button
          key={kind}
          size="sm"
          variant="outline"
          disabled={create.isPending}
          onClick={() => create.mutate({ kind }, {
            onSuccess: () => toast.success(t("run.export.queued", { kind })),
            onError: (err) => toast.error(errorDetail(err)),
          })}
        >
          {t(`run.export.create.${kind}`)}
        </Button>
      ))}
    </div>
  );
  if (exports.isPending) return <Skeleton className="m-3 h-16" />;
  if (exports.isError) return <EmptyState title={t("common.error", { detail: errorDetail(exports.error) })} />;
  const items = exports.data.items;
  return (
    <div className="flex min-h-0 flex-col gap-3 p-3 text-sm">
      {createButtons}
      <p className="text-xs text-muted-foreground">{t("run.export.keep")}</p>
      {items.length === 0 ? (
        <EmptyState title={t("run.export.none")} body={t("run.export.hint")} />
      ) : (
        <ul className="flex flex-col gap-1" data-testid="export-list">
          {items.map((e) => {
            const inFlight = e.status === "queued" || e.status === "running";
            return (
              <li key={e.name} className="flex items-center justify-between gap-2 rounded-sm border px-2 py-1 text-xs">
                <div className="flex min-w-0 items-center gap-2">
                  <JobStatusBadge status={e.status} />
                  <span className="truncate font-mono">{e.name}</span>
                  {e.status === "succeeded" && typeof e.size === "number" && (
                    <span className="text-muted-foreground">{formatBytes(e.size)}</span>
                  )}
                  {e.status === "failed" && e.error && <span className="truncate text-destructive">{e.error}</span>}
                </div>
                <div className="flex shrink-0 gap-1">
                  {e.status === "succeeded" && (
                    <Button size="sm" variant="ghost" asChild>
                      <a href={exportDownloadUrl(rid, runId, e.name)} download>{t("run.export.download")}</a>
                    </Button>
                  )}
                  <Button size="sm" variant="ghost" disabled={inFlight} onClick={() => setPendingDelete(e)}>
                    {t("common.delete")}
                  </Button>
                </div>
              </li>
            );
          })}
        </ul>
      )}
      <ConfirmDialog
        open={pendingDelete !== null}
        onOpenChange={(open) => !open && setPendingDelete(null)}
        title={t("run.export.delete.title")}
        description={pendingDelete ? t("run.export.delete.confirm", { name: pendingDelete.name }) : undefined}
        destructive
        pending={remove.isPending}
        onConfirm={() => pendingDelete && remove.mutate(pendingDelete.name, {
          onSettled: () => setPendingDelete(null),
          onError: (err) => toast.error(errorDetail(err)),
        })}
      />
    </div>
  );
}
