import { useQuery } from "@tanstack/react-query";
import { Fragment } from "react";
import { apiGet } from "@/api/client";
import { useMergeParams } from "@/api/hooks/use-params";
import { qk } from "@/api/query-keys";
import { healthSchema } from "@/api/schemas";
import { PageHeader } from "@/components/common/PageHeader";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { t } from "@/i18n";
import { formatBytes } from "@/lib/format";

// Read-only view of the backend configuration, the reference merge parameters and the hotkeys.
export function SettingsPage() {
  const health = useQuery({ queryKey: qk.health, queryFn: () => apiGet("/api/health", healthSchema) });
  const specs = useMergeParams();
  if (health.isPending || specs.isPending) return <Skeleton className="h-40 w-full" />;
  if (health.error || specs.error) return <p className="text-sm text-destructive">{t("common.error", { detail: "load failed" })}</p>;
  const h = health.data;
  const rows: Array<[string, string]> = [
    ["version", h.version],
    ["data_root", h.data_root],
    ["repo_root", h.repo_root],
    ["cpu_list", h.cpu_list ?? "(not pinned)"],
    ["gpu", h.gpu ? `${h.gpu.name} · ${h.gpu.memory_used_mib} / ${h.gpu.memory_total_mib} MiB` : "n/a"],
    ["disk", `${formatBytes(h.disk.free)} free of ${formatBytes(h.disk.total)}`],
  ];
  return (
    <div className="space-y-6">
      <PageHeader title={t("settings.title")} />
      <section>
        <h2 className="mb-2 text-sm font-semibold">{t("settings.backend")}</h2>
        <dl className="grid w-fit grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-xs">
          {rows.map(([k, v]) => (
            <Fragment key={k}>
              <dt className="font-mono text-muted-foreground">{k}</dt>
              <dd className="font-mono">{v}</dd>
            </Fragment>
          ))}
        </dl>
      </section>
      <section>
        <h2 className="mb-2 text-sm font-semibold">{t("settings.defaults")}</h2>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>name</TableHead>
              <TableHead>default</TableHead>
              <TableHead>group</TableHead>
              <TableHead>help</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {specs.data.map((p) => (
              <TableRow key={p.name}>
                <TableCell className="font-mono text-xs">{p.name}</TableCell>
                <TableCell className="font-mono text-xs">{String(p.default)}</TableCell>
                <TableCell className="text-xs">{p.group}</TableCell>
                <TableCell className="text-xs text-muted-foreground">{p.help}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </section>
      <section>
        <h2 className="mb-2 text-sm font-semibold">{t("settings.hotkeys")}</h2>
        <dl className="grid w-fit grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-xs">
          <dt className="font-mono">Ctrl+J</dt>
          <dd>{t("settings.hotkey.dock")}</dd>
        </dl>
      </section>
    </div>
  );
}
