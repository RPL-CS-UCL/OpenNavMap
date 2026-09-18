import { RefreshCw, Trash2 } from "lucide-react";
import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { toast } from "sonner";
import { errorDetail } from "@/api/client";
import { useDeleteSession, useRevalidateSession, useSession } from "@/api/hooks/use-sessions";
import { ConfirmDialog } from "@/components/common/ConfirmDialog";
import { EmptyState } from "@/components/common/EmptyState";
import { PageHeader } from "@/components/common/PageHeader";
import { StatusBadge } from "@/components/common/StatusBadge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { t } from "@/i18n";
import { SessionStatusBadge } from "@/features/regions/RegionDetailPage";
import { ValidationReportTable } from "./ValidationReportTable";

function Stat({ label, value, mono }: { label: string; value: React.ReactNode; mono?: boolean }) {
  return (
    <div className="min-w-0 rounded-sm border px-3 py-2">
      <div className="text-[11px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className={mono ? "truncate font-mono text-xs" : "text-sm tabular-nums"} title={typeof value === "string" ? value : undefined}>
        {value}
      </div>
    </div>
  );
}

export function SessionDetailPage() {
  const { rid = "", sid = "" } = useParams();
  const navigate = useNavigate();
  const session = useSession(rid, sid);
  const revalidate = useRevalidateSession(rid);
  const remove = useDeleteSession(rid);
  const [confirm, setConfirm] = useState(false);

  if (session.isPending) return <Skeleton className="h-40 w-full" />;
  if (session.error) return <p className="text-sm text-destructive">{t("common.error", { detail: session.error.message })}</p>;
  const s = session.data;

  return (
    <div>
      <PageHeader
        backTo={`/regions/${rid}`}
        title={
          <span className="flex items-center gap-2">
            {s.name}
            <SessionStatusBadge session={s} />
          </span>
        }
        actions={
          <>
            <Button
              size="sm"
              variant="outline"
              disabled={revalidate.isPending}
              onClick={() =>
                revalidate.mutate(sid, {
                  onSuccess: () => toast.success(t("session.detail.revalidated")),
                  onError: (err) => toast.error(errorDetail(err)),
                })
              }
            >
              <RefreshCw className="mr-1 h-3.5 w-3.5" />
              {t("session.detail.revalidate")}
            </Button>
            <Button size="sm" variant="ghost" aria-label={t("common.delete")} onClick={() => setConfirm(true)}>
              <Trash2 className="h-3.5 w-3.5" />
            </Button>
          </>
        }
      />
      <div className="mb-4 grid grid-cols-4 gap-2">
        <Stat label={t("session.detail.frames")} value={s.num_frames} />
        <Stat label={t("session.detail.source")} value={<StatusBadge tone="muted">{s.source}</StatusBadge>} />
        <Stat label={t("session.detail.descriptorDim")} value={s.validation?.descriptor_dim ?? "–"} />
        <Stat label={t("session.detail.path")} value={s.path} mono />
      </div>
      <Tabs defaultValue="files">
        <TabsList>
          <TabsTrigger value="files">{t("session.tabs.files")}</TabsTrigger>
          <TabsTrigger value="frames">{t("session.tabs.frames")}</TabsTrigger>
          <TabsTrigger value="trajectory">{t("session.tabs.trajectory")}</TabsTrigger>
          <TabsTrigger value="jobs">{t("session.tabs.jobs")}</TabsTrigger>
        </TabsList>
        <TabsContent value="files" className="pt-3">
          {s.validation ? <ValidationReportTable report={s.validation} /> : <EmptyState title={t("validation.none")} />}
        </TabsContent>
        <TabsContent value="frames" className="pt-3">
          <EmptyState title={t("session.tabs.frames")} body={t("session.comingSoon")} />
        </TabsContent>
        <TabsContent value="trajectory" className="pt-3">
          <EmptyState title={t("session.tabs.trajectory")} body={t("session.comingSoon")} />
        </TabsContent>
        <TabsContent value="jobs" className="pt-3">
          <EmptyState title={t("session.tabs.jobs")} body={t("session.comingSoon")} />
        </TabsContent>
      </Tabs>
      <ConfirmDialog
        open={confirm}
        onOpenChange={setConfirm}
        title={t("common.delete")}
        description={t("common.confirmDelete", { name: s.name })}
        destructive
        pending={remove.isPending}
        onConfirm={() =>
          remove.mutate(sid, {
            onSuccess: () => {
              toast.success(t("sessions.deleted", { name: s.name }));
              navigate(`/regions/${rid}`);
            },
            onError: (err) => toast.error(errorDetail(err)),
          })
        }
      />
    </div>
  );
}
