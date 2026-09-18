import type { ParamSpec, Run, RunParent, Session } from "@/api/types";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { t } from "@/i18n";
import { formatDuration } from "@/lib/format";
import { diffParams } from "./param-schema";

const DEFAULT_STEP_SECONDS = 6 * 60;

/** Per-step wall time from the most recent successful run, or a 6-minute default. */
export function estimateStepSeconds(runs: Run[]): number {
  const done = runs.filter((r) => r.status === "succeeded" && r.finished_at && r.num_steps_expected > 0);
  if (done.length === 0) return DEFAULT_STEP_SECONDS;
  const last = done[0];
  const seconds = (new Date(last.finished_at as string).getTime() - new Date(last.created_at).getTime()) / 1000;
  return seconds > 0 ? seconds / last.num_steps_expected : DEFAULT_STEP_SECONDS;
}

interface Props {
  specs: ParamSpec[];
  sessions: Session[];
  order: string[];
  mode: "merge" | "append";
  parent: RunParent | null;
  params: Record<string, unknown>;
  name: string;
  runs: Run[];
  onName: (n: string) => void;
}

export function ReviewStep({ specs, sessions, order, mode, parent, params, name, runs, onName }: Props) {
  const byId = new Map(sessions.map((s) => [s.id, s]));
  const changes = diffParams(specs, params);
  const perStep = estimateStepSeconds(runs);
  return (
    <div className="space-y-4 text-sm">
      <div className="grid w-96 gap-1">
        <Label htmlFor="run-name" className="text-xs">
          {t("wizard.name")}
        </Label>
        <Input id="run-name" className="h-8" value={name} onChange={(e) => onName(e.target.value)} />
      </div>
      <section>
        <h3 className="mb-1 font-semibold">{order.length === 1 ? t("wizard.review.session") : t("wizard.review.sessions", { count: order.length })}</h3>
        <ol className="list-decimal pl-5 font-mono text-xs">
          {order.map((id) => (
            <li key={id}>{byId.get(id)?.name ?? id}</li>
          ))}
        </ol>
      </section>
      <section>
        <h3 className="mb-1 font-semibold">{t("wizard.review.mode")}</h3>
        <p className="text-xs">{mode === "merge" ? t("wizard.mode.fresh") : `${t("wizard.mode.append")}${parent ? ` · ${parent.run_id} · ${parent.step_index}` : ""}`}</p>
      </section>
      <section>
        <h3 className="mb-1 font-semibold">{t("wizard.review.changes")}</h3>
        {changes.length === 0 ? (
          <p className="text-xs text-muted-foreground">{t("wizard.review.noChanges")}</p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>name</TableHead>
                <TableHead>default</TableHead>
                <TableHead>value</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {changes.map((c) => (
                <TableRow key={c.name}>
                  <TableCell className="font-mono text-xs">{c.name}</TableCell>
                  <TableCell className="font-mono text-xs">{String(c.from)}</TableCell>
                  <TableCell className="font-mono text-xs">{String(c.to)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </section>
      <p className="text-xs text-muted-foreground">{t("wizard.review.eta", { eta: formatDuration(perStep * order.length), perStep: formatDuration(perStep) })}</p>
    </div>
  );
}
