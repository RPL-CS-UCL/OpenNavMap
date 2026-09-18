import { AlertTriangle, Ban, Check, X } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type { FileCheck, ValidationReport } from "@/api/types";
import { StatusBadge } from "@/components/common/StatusBadge";
import type { Tone } from "@/components/common/StatusBadge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { t } from "@/i18n";

const statusStyle: Record<FileCheck["status"], { tone: Tone; icon: LucideIcon }> = {
  ok: { tone: "ok", icon: Check },
  missing: { tone: "error", icon: X },
  incomplete: { tone: "warn", icon: AlertTriangle },
  invalid: { tone: "error", icon: Ban },
};

export function ValidationReportTable({ report }: { report: ValidationReport }) {
  return (
    <div className="space-y-3">
      {report.errors.length > 0 && (
        <Alert variant="destructive">
          <AlertTitle>{t("validation.errors")}</AlertTitle>
          <AlertDescription>
            <ul className="list-disc pl-4">
              {report.errors.map((e) => (
                <li key={e}>{e}</li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      )}
      {report.warnings.length > 0 && (
        <Alert>
          <AlertTitle>{t("validation.warnings")}</AlertTitle>
          <AlertDescription>
            <ul className="list-disc pl-4">
              {report.warnings.map((w) => (
                <li key={w}>{w}</li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      )}
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>{t("validation.col.file")}</TableHead>
            <TableHead>{t("validation.col.required")}</TableHead>
            <TableHead>{t("validation.col.status")}</TableHead>
            <TableHead className="text-right">{t("validation.col.lines")}</TableHead>
            <TableHead>{t("validation.col.detail")}</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {report.files.map((f) => {
            const { tone, icon } = statusStyle[f.status];
            // Missing optional files are the normal case; status text keeps the meaning without color.
            const effectiveTone: Tone = f.status === "missing" && !f.required ? "muted" : tone;
            return (
              <TableRow key={f.name}>
                <TableCell className="font-mono text-xs">{f.name}</TableCell>
                <TableCell className="text-xs text-muted-foreground">
                  {f.required ? t("validation.required.yes") : t("validation.required.no")}
                </TableCell>
                <TableCell>
                  <StatusBadge tone={effectiveTone} icon={icon}>
                    {t(`validation.status.${f.status}`)}
                  </StatusBadge>
                </TableCell>
                <TableCell className="text-right font-mono text-xs tabular-nums">{f.lines ?? "–"}</TableCell>
                <TableCell className="text-xs text-muted-foreground">{f.detail}</TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}
