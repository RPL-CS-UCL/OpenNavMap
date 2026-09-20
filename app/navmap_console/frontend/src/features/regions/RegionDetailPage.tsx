import { Ban, Check, FolderDown, HelpCircle, Plus, Trash2, X } from "lucide-react";
import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { toast } from "sonner";
import { errorDetail } from "@/api/client";
import { useDeleteRegion, useRegion } from "@/api/hooks/use-regions";
import { useDeleteSession, useSessions } from "@/api/hooks/use-sessions";
import { useRuns } from "@/api/hooks/use-runs";
import type { Session } from "@/api/types";
import { ConfirmDialog } from "@/components/common/ConfirmDialog";
import { EmptyState } from "@/components/common/EmptyState";
import { PageHeader } from "@/components/common/PageHeader";
import { StatusBadge } from "@/components/common/StatusBadge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { JobStatusBadge } from "@/features/jobs/job-status";
import { ImportRunDialog } from "@/features/runs/ImportRunDialog";
import { t } from "@/i18n";
import { formatDateTime } from "@/lib/format";

export function SessionStatusBadge({ session }: { session: Session }) {
  if (!session.validation) return <StatusBadge tone="muted" icon={HelpCircle}>{t("sessions.status.unknown")}</StatusBadge>;
  return session.validation.ok ? (
    <StatusBadge tone="ok" icon={Check}>{t("sessions.status.ok")}</StatusBadge>
  ) : (
    <StatusBadge tone="error" icon={X}>{t("sessions.status.failed")}</StatusBadge>
  );
}

function flag(on: boolean, label: string) {
  return (
    <span className="inline-flex items-center gap-0.5" title={`${label}: ${on ? "yes" : "no"}`}>
      {on ? <Check className="h-3 w-3" /> : <Ban className="h-3 w-3 text-muted-foreground/60" />}
      <span className={on ? "" : "text-muted-foreground/60"}>{label}</span>
    </span>
  );
}

export function RegionDetailPage() {
  const { rid = "" } = useParams();
  const navigate = useNavigate();
  const region = useRegion(rid);
  const sessions = useSessions(rid);
  const runs = useRuns(rid);
  const deleteRegion = useDeleteRegion();
  const deleteSession = useDeleteSession(rid);
  const [confirmRegion, setConfirmRegion] = useState(false);
  const [sessionToDelete, setSessionToDelete] = useState<Session | null>(null);
  const [importOpen, setImportOpen] = useState(false);

  if (region.isPending) return <Skeleton className="h-40 w-full" />;
  if (region.error) return <p className="text-sm text-destructive">{t("common.error", { detail: region.error.message })}</p>;
  const r = region.data;

  return (
    <div>
      <PageHeader
        backTo="/regions"
        title={r.name}
        description={r.description || `${r.vpr.method}/${r.vpr.backbone}/${r.vpr.dim} · ${r.image_size.join("×")}`}
        actions={
          <>
            <Button asChild size="sm" variant="outline">
              <Link to={`/regions/${rid}/sessions/new`}>
                <Plus className="mr-1 h-3.5 w-3.5" />
                {t("region.newSession")}
              </Link>
            </Button>
            <Button size="sm" variant="outline" onClick={() => setImportOpen(true)}>
              <FolderDown className="mr-1 h-3.5 w-3.5" />
              {t("region.importRun")}
            </Button>
            <Button asChild size="sm">
              <Link to={`/regions/${rid}/runs/new`}>{t("region.newRun")}</Link>
            </Button>
            <Button size="sm" variant="ghost" aria-label={t("common.delete")} onClick={() => setConfirmRegion(true)}>
              <Trash2 className="h-3.5 w-3.5" />
            </Button>
          </>
        }
      />

      <div className="grid grid-cols-[1fr_320px] gap-4">
        <div className="space-y-6">
          <section>
            <h2 className="mb-2 text-sm font-semibold">{t("region.sessions")}</h2>
            {sessions.isPending && <Skeleton className="h-20 w-full" />}
            {sessions.data && sessions.data.length === 0 && <EmptyState title={t("sessions.empty")} />}
            {sessions.data && sessions.data.length > 0 && (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>{t("sessions.col.name")}</TableHead>
                    <TableHead className="text-right">{t("sessions.col.frames")}</TableHead>
                    <TableHead>{t("sessions.col.source")}</TableHead>
                    <TableHead>{t("sessions.col.flags")}</TableHead>
                    <TableHead>{t("sessions.col.status")}</TableHead>
                    <TableHead />
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {sessions.data.map((s) => (
                    <TableRow key={s.id} aria-label={s.name}>
                      <TableCell>
                        <Link to={`/regions/${rid}/sessions/${s.id}`} className="font-medium hover:underline">
                          {s.name}
                        </Link>
                        <div className="truncate font-mono text-[11px] text-muted-foreground" title={s.path}>
                          {s.path}
                        </div>
                      </TableCell>
                      <TableCell className="text-right tabular-nums">{s.num_frames}</TableCell>
                      <TableCell>
                        <StatusBadge tone="muted">{s.source}</StatusBadge>
                      </TableCell>
                      <TableCell className="space-x-2 text-xs">
                        {flag(s.has_gt, "GT")}
                        {flag(s.has_gps, "GPS")}
                        {flag(s.has_iqa, "IQA")}
                      </TableCell>
                      <TableCell>
                        <SessionStatusBadge session={s} />
                      </TableCell>
                      <TableCell className="text-right">
                        <Button
                          size="sm"
                          variant="ghost"
                          className="h-7 w-7 p-0"
                          aria-label={`${t("common.delete")} ${s.name}`}
                          onClick={() => setSessionToDelete(s)}
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </section>

          <section>
            <h2 className="mb-2 text-sm font-semibold">{t("region.runs")}</h2>
            {runs.isPending && <Skeleton className="h-20 w-full" />}
            {runs.data && runs.data.length === 0 && <EmptyState title={t("region.runs.empty")} />}
            {runs.data && runs.data.length > 0 && (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>{t("runs.col.name")}</TableHead>
                    <TableHead>{t("runs.col.kind")}</TableHead>
                    <TableHead>{t("runs.col.status")}</TableHead>
                    <TableHead className="text-right">{t("runs.col.steps")}</TableHead>
                    <TableHead>{t("runs.col.created")}</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {runs.data.map((run) => (
                    <TableRow key={run.id} aria-label={run.name || run.id}>
                      <TableCell>
                        <Link to={`/regions/${rid}/runs/${run.id}`} className="font-medium hover:underline">
                          {run.name || run.id}
                        </Link>
                        <div className="font-mono text-[11px] text-muted-foreground">{run.id}</div>
                      </TableCell>
                      <TableCell>
                        <StatusBadge tone="muted">{run.kind}</StatusBadge>
                      </TableCell>
                      <TableCell>
                        <JobStatusBadge status={run.status} />
                      </TableCell>
                      <TableCell className="text-right font-mono tabular-nums">
                        {`${run.last_step_index === null ? "-" : run.last_step_index + 1} / ${run.num_steps_expected}`}
                      </TableCell>
                      <TableCell className="text-xs text-muted-foreground">{formatDateTime(run.created_at)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </section>
        </div>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">{t("region.finalMap")}</CardTitle>
          </CardHeader>
          <CardContent className="text-xs text-muted-foreground">
            {r.head ? (
              <div className="space-y-1">
                <Link to={`/regions/${rid}/runs/${r.head.run_id}`} className="block font-mono text-foreground hover:underline">
                  {t("regions.map.head", { run: r.head.run_id, step: r.head.step_index })}
                </Link>
                <div>{t("region.finalMap.sessions", { count: r.head.session_ids.length })}</div>
              </div>
            ) : (
              t("region.finalMap.none")
            )}
          </CardContent>
        </Card>
      </div>

      <ConfirmDialog
        open={confirmRegion}
        onOpenChange={setConfirmRegion}
        title={t("common.delete")}
        description={t("common.confirmDelete", { name: r.name })}
        destructive
        pending={deleteRegion.isPending}
        onConfirm={() =>
          deleteRegion.mutate(rid, {
            onSuccess: () => {
              toast.success(t("region.deleted", { name: r.name }));
              navigate("/regions");
            },
            onError: (err) => toast.error(errorDetail(err)),
          })
        }
      />
      <ImportRunDialog open={importOpen} onOpenChange={setImportOpen} />
      <ConfirmDialog
        open={sessionToDelete !== null}
        onOpenChange={(o) => !o && setSessionToDelete(null)}
        title={t("common.delete")}
        description={sessionToDelete ? t("common.confirmDelete", { name: sessionToDelete.name }) : ""}
        destructive
        pending={deleteSession.isPending}
        onConfirm={() => {
          const s = sessionToDelete;
          if (!s) return;
          deleteSession.mutate(s.id, {
            onSuccess: () => {
              toast.success(t("sessions.deleted", { name: s.name }));
              setSessionToDelete(null);
            },
            onError: (err) => toast.error(errorDetail(err)),
          });
        }}
      />
    </div>
  );
}
