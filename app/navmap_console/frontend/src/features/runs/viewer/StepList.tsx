import type { StepSummary } from "@/api/types";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { t } from "@/i18n";
import { cn } from "@/lib/utils";
import { useSceneStore } from "@/stores/scene-store";

interface Props { steps: StepSummary[] | undefined; pending: boolean }

export function StepList({ steps, pending }: Props) {
  const step = useSceneStore((s) => s.step);
  const maxStep = useSceneStore((s) => s.maxStep);
  const follow = useSceneStore((s) => s.follow);
  const setStep = useSceneStore((s) => s.setStep);
  const setFollow = useSceneStore((s) => s.setFollow);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="min-h-0 flex-1 overflow-auto">
        {pending && [0, 1, 2].map((i) => <Skeleton key={i} className="mx-2 mb-1 h-7" />)}
        {steps?.map((s, k) => {
          // PGO error proxy for ATE (M5 swaps in the real number): worse than the previous step.
          const degraded = s.pgo_error_final !== null && k > 0 && (steps[k - 1].pgo_error_final ?? 0) < s.pgo_error_final;
          const split = s.component_sizes.length > 1;
          const allRejected = s.loops.new > 0 && s.loops.rejected_new === s.loops.new;
          return (
            <button
              key={k}
              type="button"
              aria-label={String(k)}
              aria-current={step === k ? "step" : undefined}
              onClick={() => setStep(k)}
              className={cn(
                "flex w-full items-center gap-2 px-2 py-1 text-left text-xs hover:bg-accent",
                step === k && "bg-accent font-medium",
              )}
            >
              <span className="w-6 shrink-0 font-mono tabular-nums">{k}</span>
              <span className="min-w-0 flex-1 truncate font-mono text-[11px] text-muted-foreground">{s.session_id}</span>
              {split && <span title={t("viewer.steps.marker.split", { n: s.component_sizes.length })}>⚠ {s.component_sizes.length}</span>}
              {degraded && <span title={t("viewer.steps.marker.degraded")}>↓</span>}
              {allRejected && (
                <span
                  data-testid={`all-rejected-${k}`}
                  title={t("viewer.steps.marker.allRejected")}
                  className="h-2 w-2 shrink-0 rounded-full bg-destructive"
                />
              )}
              <span className="w-10 shrink-0 text-right font-mono tabular-nums">{s.num_nodes}</span>
            </button>
          );
        })}
      </div>
      <div className="flex items-center justify-between gap-2 border-t px-2 py-1.5">
        <div className="flex items-center gap-1.5 text-xs">
          <Switch checked={follow} onCheckedChange={setFollow} aria-label={t("viewer.steps.follow")} />
          <span>{t("viewer.steps.follow")}</span>
        </div>
        {!follow && step !== null && maxStep > step && (
          <span className="rounded-sm bg-accent px-1.5 py-0.5 text-[11px] text-accent-foreground">
            {t("viewer.steps.newSteps", { count: maxStep - step })}
          </span>
        )}
      </div>
    </div>
  );
}
