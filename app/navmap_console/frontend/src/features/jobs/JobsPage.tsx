import { useState } from "react";
import { Link } from "react-router-dom";
import { useJobs } from "@/api/hooks/use-jobs";
import { EmptyState } from "@/components/common/EmptyState";
import { PageHeader } from "@/components/common/PageHeader";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { t } from "@/i18n";
import { formatDateTime, formatDuration } from "@/lib/format";
import { JobStatusBadge, progressText } from "./job-status";

const FILTERS = { all: "", active: "queued,running", done: "succeeded,failed,cancelled,orphaned" } as const;
type Filter = keyof typeof FILTERS;

export function durationOf(startedAt: string | null, finishedAt: string | null): string {
  if (!startedAt) return "";
  const end = finishedAt ? new Date(finishedAt).getTime() : Date.now();
  return formatDuration((end - new Date(startedAt).getTime()) / 1000);
}

export function JobsPage() {
  const [filter, setFilter] = useState<Filter>("all");
  const jobs = useJobs(FILTERS[filter]);
  return (
    <div>
      <PageHeader
        title={t("jobs.title")}
        actions={
          <ToggleGroup type="single" size="sm" value={filter} onValueChange={(v) => v && setFilter(v as Filter)}>
            <ToggleGroupItem value="all">{t("jobs.filter.all")}</ToggleGroupItem>
            <ToggleGroupItem value="active">{t("jobs.filter.active")}</ToggleGroupItem>
            <ToggleGroupItem value="done">{t("jobs.filter.done")}</ToggleGroupItem>
          </ToggleGroup>
        }
      />
      {jobs.isPending && <Skeleton className="h-24 w-full" />}
      {jobs.data && jobs.data.length === 0 && <EmptyState title={t("jobs.empty")} />}
      {jobs.data && jobs.data.length > 0 && (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>{t("jobs.col.id")}</TableHead>
              <TableHead>{t("jobs.col.kind")}</TableHead>
              <TableHead>{t("jobs.col.status")}</TableHead>
              <TableHead>{t("jobs.col.progress")}</TableHead>
              <TableHead>{t("jobs.col.stage")}</TableHead>
              <TableHead>{t("jobs.col.started")}</TableHead>
              <TableHead className="text-right">{t("jobs.col.duration")}</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {jobs.data.map((j) => (
              <TableRow key={j.id} aria-label={j.id}>
                <TableCell>
                  <Link to={`/jobs/${j.id}`} className="font-mono text-xs hover:underline">
                    {j.id}
                  </Link>
                </TableCell>
                <TableCell>{j.kind}</TableCell>
                <TableCell>
                  <JobStatusBadge status={j.status} />
                </TableCell>
                <TableCell className="tabular-nums">{progressText(j.progress)}</TableCell>
                <TableCell className="text-xs text-muted-foreground">{j.progress.stage ?? ""}</TableCell>
                <TableCell className="text-xs">{j.started_at ? formatDateTime(j.started_at) : ""}</TableCell>
                <TableCell className="text-right font-mono text-xs tabular-nums">
                  {durationOf(j.started_at, j.finished_at)}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </div>
  );
}
