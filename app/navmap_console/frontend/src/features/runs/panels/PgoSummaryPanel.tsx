import type { Scene } from "@/api/scene-bundle";
import type { StepSummary } from "@/api/types";
import { EmptyState } from "@/components/common/EmptyState";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { t } from "@/i18n";
import { useSceneStore } from "@/stores/scene-store";

/** Ten equal buckets over [0, 1]; the last one is right-closed so 1.0 lands in bin 9. NaN/非有限值跳过. */
export function weightHistogram(weights: Float32Array | null, bins = 10): number[] {
  const out = new Array<number>(bins).fill(0);
  if (!weights) return out;
  for (const w of weights) {
    if (!Number.isFinite(w)) continue;
    out[Math.min(bins - 1, Math.floor(w * bins))] += 1;
  }
  return out;
}

const fmt = (v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(2));

function Stat({ label, value, warn }: { label: string; value: string; warn?: boolean }) {
  return (
    <div className="flex flex-col gap-0.5 rounded-sm border p-1.5">
      <span className="text-muted-foreground">{label}</span>
      <span className={`font-mono text-sm ${warn ? "text-amber-600" : ""}`}>{value}</span>
    </div>
  );
}

/** PGO outcome of one merge step: error delta, loop stats, weight histogram, ghost-layer controls. */
export function PgoSummaryPanel({ summary, scene }: { summary: StepSummary | undefined; scene: Scene | null }) {
  const morphThreshold = useSceneStore((s) => s.morphThreshold);
  const ghostOn = useSceneStore((s) => s.layers.ghost);
  if (!summary) return <EmptyState title={t("panel.pgo.noStep")} />;
  const hist = weightHistogram(scene?.loopWeight ?? null, 10);
  const max = Math.max(1, ...hist);
  const overturnRatio = summary.loops.hist > 0 ? summary.loops.overturned_hist / summary.loops.hist : 0;
  const delta = summary.pgo_error_initial === null || summary.pgo_error_final === null
    ? null : summary.pgo_error_final - summary.pgo_error_initial;
  const improved = delta !== null && delta < 0;
  return (
    <div className="flex h-full min-h-0 flex-col gap-1.5 overflow-auto p-2 text-xs">
      <div className="flex items-center justify-between gap-2">
        <span className="text-muted-foreground">{t("panel.pgo.error")}</span>
        <span className="font-mono">
          {summary.pgo_error_initial === null || summary.pgo_error_final === null
            ? "–"
            : `${fmt(summary.pgo_error_initial)} → ${fmt(summary.pgo_error_final)}`}
          {delta !== null && (
            <span className={improved ? "text-green-600" : "text-amber-600"}> {improved ? "↓" : "↑"}{fmt(Math.abs(delta))}</span>
          )}
        </span>
      </div>
      {overturnRatio > 0.5 && (
        <div className="text-amber-600">{t("panel.pgo.overturnedWarn")}</div>
      )}
      <div className="grid grid-cols-3 gap-1.5">
        <Stat label={t("panel.pgo.accepted")} value={String(summary.loops.accepted)} />
        <Stat label={t("panel.pgo.rejectedNew")} value={String(summary.loops.rejected_new)} />
        <Stat label={t("panel.pgo.overturnedHist")} value={`${summary.loops.overturned_hist} / ${summary.loops.hist}`} warn={overturnRatio > 0.5} />
      </div>
      <div className="grid grid-cols-2 gap-1.5">
        <Stat label={t("panel.pgo.components")} value={String(summary.component_sizes.length)} warn={summary.component_sizes.length > 1} />
        <Stat label={t("panel.pgo.moved")} value={`${summary.num_moved} · ${summary.max_displacement.toFixed(3)} m`} />
      </div>
      <div className="flex flex-col gap-0.5">
        <span className="text-muted-foreground">{t("panel.pgo.histogram")}</span>
        <div className="relative flex h-10 items-end gap-px border-b">
          {hist.map((c, i) => (
            <div key={i} className="flex-1 rounded-t-sm bg-primary/70" style={{ height: `${Math.round((c / max) * 100)}%` }} title={String(c)} />
          ))}
          <span className="absolute bottom-0 left-1/2 top-0 w-px bg-amber-500" title={t("panel.pgo.histogramRef")} />
        </div>
      </div>
      <div className="flex items-center justify-between gap-2">
        <span className="text-muted-foreground">{t("panel.pgo.threshold")}</span>
        <Slider
          className="w-32"
          min={0.01}
          max={0.5}
          step={0.01}
          value={[morphThreshold]}
          onValueChange={([v]) => useSceneStore.getState().setMorphThreshold(v)}
          aria-label={t("panel.pgo.threshold")}
        />
      </div>
      <div className="flex items-center justify-between gap-2">
        <label htmlFor="pgo-ghost" className="text-muted-foreground">{t("panel.pgo.ghost")}</label>
        <Switch
          id="pgo-ghost"
          checked={ghostOn}
          onCheckedChange={() => useSceneStore.getState().toggleLayer("ghost")}
        />
      </div>
      <div className="flex gap-1">
        <Badge variant="secondary">{t("panel.pgo.histogramRef")} 0.5</Badge>
      </div>
    </div>
  );
}
