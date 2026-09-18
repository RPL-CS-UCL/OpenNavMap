import type { Region, Run, RunParent } from "@/api/types";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { t } from "@/i18n";

type Mode = "merge" | "append";

interface Props {
  region: Region;
  runs: Run[];
  mode: Mode;
  parent: RunParent | null;
  onMode: (m: Mode) => void;
  onParent: (p: RunParent | null) => void;
}

export function ModeStep({ region, runs, mode, parent, onMode, onParent }: Props) {
  const head = region.head;
  const candidates = runs.filter((r) => r.status === "succeeded" && r.last_step_index !== null);
  const parentValue = parent ? `${parent.run_id}:${parent.step_index}` : "head";
  return (
    <div className="space-y-4">
      <RadioGroup value={mode} onValueChange={(v) => onMode(v as Mode)}>
        <div className="flex items-start gap-3 rounded-sm border p-3">
          <RadioGroupItem value="merge" id="mode-merge" aria-label={t("wizard.mode.fresh")} />
          <Label htmlFor="mode-merge" className="space-y-1">
            <div className="text-sm font-medium">{t("wizard.mode.fresh")}</div>
            <div className="text-xs text-muted-foreground">{t("wizard.mode.freshHint")}</div>
          </Label>
        </div>
        <div className="flex items-start gap-3 rounded-sm border p-3">
          <RadioGroupItem value="append" id="mode-append" disabled={!head} aria-label={t("wizard.mode.append")} />
          <Label htmlFor="mode-append" className="space-y-1">
            <div className="text-sm font-medium">{t("wizard.mode.append")}</div>
            <div className="text-xs text-muted-foreground">
              {head ? t("wizard.mode.appendHint", { run: head.run_id, step: head.step_index }) : t("wizard.mode.appendNone")}
            </div>
          </Label>
        </div>
      </RadioGroup>
      {mode === "append" && head && (
        <div className="space-y-2">
          <Label className="text-xs">{t("wizard.mode.otherParent")}</Label>
          <Select
            value={parentValue}
            onValueChange={(v) => onParent(v === "head" ? null : { run_id: v.split(":")[0], step_index: Number(v.split(":")[1]) })}
          >
            <SelectTrigger className="h-8 w-96 font-mono text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="head">
                {t("wizard.mode.parentHead")} · {head.run_id} · {head.step_index}
              </SelectItem>
              {candidates.map((r) => (
                <SelectItem key={r.id} value={`${r.id}:${r.last_step_index}`}>
                  {r.name || r.id} · step {r.last_step_index}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Alert className="border-amber-600/40 bg-amber-500/10">
            <AlertDescription className="text-xs">{t("wizard.mode.warning")}</AlertDescription>
          </Alert>
        </div>
      )}
    </div>
  );
}
