import { AlertTriangle } from "lucide-react";
import type { StepRecord } from "@/api/types";
import { StatusBadge } from "@/components/common/StatusBadge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { durationOf } from "@/features/jobs/JobsPage";
import { t } from "@/i18n";

interface Props {
  steps: StepRecord[];
  expected: number;
  startStep: number;
  sessionNames?: Record<string, string>;
}

const tone = { done: "ok", running: "warn", failed: "error" } as const;

function num(v: number | null | undefined, digits = 0): string {
  return v === null || v === undefined ? "-" : v.toFixed(digits);
}

// One row per expected step; rows without a record yet render as "pending" so the table keeps its final shape.
export function StepsTable({ steps, expected, startStep, sessionNames = {} }: Props) {
  const byIndex = new Map(steps.map((s) => [s.index, s]));
  const rows = Array.from({ length: Math.max(expected, steps.length) }, (_, k) => startStep + k);
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>{t("run.steps.col.index")}</TableHead>
          <TableHead>{t("run.steps.col.session")}</TableHead>
          <TableHead>{t("run.steps.col.status")}</TableHead>
          <TableHead className="text-right">{t("run.steps.col.nodes")}</TableHead>
          <TableHead className="text-right">{t("run.steps.col.components")}</TableHead>
          <TableHead className="text-right">{t("run.steps.col.registry")}</TableHead>
          <TableHead className="text-right">{t("run.steps.col.pgo")}</TableHead>
          <TableHead className="text-right">{t("run.steps.col.duration")}</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {rows.map((index) => {
          const s = byIndex.get(index);
          const split = s?.components !== null && s?.components !== undefined && s.components > 1;
          return (
            <TableRow key={index} aria-label={`step ${index}`}>
              <TableCell className="font-mono tabular-nums">{index}</TableCell>
              <TableCell className="font-mono text-xs">{s ? (sessionNames[s.session_id] ?? s.session_id) : ""}</TableCell>
              <TableCell>
                {s ? <StatusBadge tone={tone[s.status]}>{s.status}</StatusBadge> : <StatusBadge tone="muted">{t("run.steps.pending")}</StatusBadge>}
              </TableCell>
              <TableCell className="text-right font-mono tabular-nums">{s ? `${num(s.odom_nodes)} / ${num(s.covis_nodes)}` : ""}</TableCell>
              <TableCell className="text-right font-mono tabular-nums">
                {split ? (
                  <span className="inline-flex items-center gap-1 text-amber-700 dark:text-amber-300" title={t("run.steps.split", { n: s!.components! })}>
                    <AlertTriangle className="h-3 w-3" />
                    {s!.components}
                  </span>
                ) : (
                  num(s?.components)
                )}
              </TableCell>
              <TableCell className="text-right font-mono tabular-nums">{num(s?.registry_edges)}</TableCell>
              <TableCell className="text-right font-mono tabular-nums">{s ? `${num(s.pgo_error_initial, 3)} / ${num(s.pgo_error_final, 3)}` : ""}</TableCell>
              <TableCell className="text-right font-mono tabular-nums">{s ? durationOf(s.started_at, s.finished_at) : ""}</TableCell>
            </TableRow>
          );
        })}
      </TableBody>
    </Table>
  );
}
