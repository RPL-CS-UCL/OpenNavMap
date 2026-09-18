import { Map, Plus } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import { useRegions } from "@/api/hooks/use-regions";
import { EmptyState } from "@/components/common/EmptyState";
import { PageHeader } from "@/components/common/PageHeader";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { t } from "@/i18n";
import { formatDateTime } from "@/lib/format";
import { NewRegionDialog } from "./NewRegionDialog";

export function RegionsPage() {
  const { data, isPending, error } = useRegions();
  const [open, setOpen] = useState(false);

  const newButton = (
    <Button size="sm" onClick={() => setOpen(true)}>
      <Plus className="mr-1 h-3.5 w-3.5" />
      {t("regions.new")}
    </Button>
  );

  return (
    <div>
      <PageHeader title={t("regions.title")} actions={newButton} />
      {isPending && <Skeleton className="h-24 w-full" />}
      {error && <p className="text-sm text-destructive">{t("common.error", { detail: error.message })}</p>}
      {data && data.length === 0 && (
        <EmptyState icon={Map} title={t("regions.empty.title")} body={t("regions.empty.body")} action={newButton} />
      )}
      {data && data.length > 0 && (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>{t("regions.col.name")}</TableHead>
              <TableHead className="text-right">{t("regions.col.sessions")}</TableHead>
              <TableHead className="text-right">{t("regions.col.runs")}</TableHead>
              <TableHead>{t("regions.col.map")}</TableHead>
              <TableHead>{t("regions.col.created")}</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {data.map((r) => (
              <TableRow key={r.id} aria-label={r.name}>
                <TableCell>
                  <Link to={`/regions/${r.id}`} className="font-medium hover:underline">
                    {r.name}
                  </Link>
                  {r.description && <div className="text-xs text-muted-foreground">{r.description}</div>}
                </TableCell>
                <TableCell className="text-right tabular-nums">{r.session_count ?? 0}</TableCell>
                <TableCell className="text-right tabular-nums">{r.run_count ?? 0}</TableCell>
                <TableCell className="font-mono text-xs">
                  {r.head ? t("regions.map.head", { run: r.head.run_id, step: r.head.step_index }) : t("regions.map.none")}
                </TableCell>
                <TableCell className="text-xs text-muted-foreground">{formatDateTime(r.created_at)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
      <NewRegionDialog open={open} onOpenChange={setOpen} />
    </div>
  );
}
