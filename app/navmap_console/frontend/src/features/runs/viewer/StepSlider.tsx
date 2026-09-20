import { ChevronLeft, ChevronRight, SkipBack, SkipForward } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { t } from "@/i18n";
import { useSceneStore } from "@/stores/scene-store";

const SPEEDS = ["0.5", "1", "2", "4"];

export function StepSlider({ hasPrePgo }: { hasPrePgo: boolean }) {
  const step = useSceneStore((s) => s.step) ?? 0;
  const maxStep = useSceneStore((s) => s.maxStep);
  const speed = useSceneStore((s) => s.speed);
  const morph = useSceneStore((s) => s.morph);
  const setStep = useSceneStore((s) => s.setStep);
  const stepBy = useSceneStore((s) => s.stepBy);
  const jumpToLatest = useSceneStore((s) => s.jumpToLatest);
  const setSpeed = useSceneStore((s) => s.setSpeed);
  const setMorph = useSceneStore((s) => s.setMorph);

  return (
    <div className="flex items-center gap-3 border-t px-2 py-1.5">
      <div className="flex items-center gap-0.5">
        <Button variant="ghost" size="icon" className="h-7 w-7" aria-label={t("viewer.slider.first")} disabled={maxStep < 0} onClick={() => setStep(0)}>
          <SkipBack />
        </Button>
        <Button variant="ghost" size="icon" className="h-7 w-7" aria-label={t("viewer.slider.prev")} disabled={maxStep < 0} onClick={() => stepBy(-1)}>
          <ChevronLeft />
        </Button>
        <Button variant="ghost" size="icon" className="h-7 w-7" aria-label={t("viewer.slider.next")} disabled={maxStep < 0} onClick={() => stepBy(1)}>
          <ChevronRight />
        </Button>
        <Button variant="ghost" size="icon" className="h-7 w-7" aria-label={t("viewer.slider.latest")} disabled={maxStep < 0} onClick={jumpToLatest}>
          <SkipForward />
        </Button>
      </div>
      <Slider
        className="min-w-24 flex-1"
        value={[step]}
        min={0}
        max={Math.max(1, maxStep)}
        disabled={maxStep < 0}
        onValueChange={([v]) => setStep(v)}
      />
      <span className="w-12 shrink-0 font-mono text-xs tabular-nums">
        {step} / {maxStep}
      </span>
      <ToggleGroup type="single" value={String(speed)} onValueChange={(v) => v && setSpeed(Number(v))} className="gap-0">
        {SPEEDS.map((s) => (
          <ToggleGroupItem key={s} value={s} className="h-6 px-1.5 text-[11px]">
            ×{s}
          </ToggleGroupItem>
        ))}
      </ToggleGroup>
      <div className="flex w-36 shrink-0 items-center gap-1.5">
        <Slider
          value={[morph]}
          min={0}
          max={1}
          step={0.01}
          disabled={!hasPrePgo}
          title={hasPrePgo ? undefined : t("viewer.slider.morphDisabled")}
          aria-label={t("viewer.slider.morph")}
          onValueChange={([v]) => setMorph(v)}
        />
        {!hasPrePgo && <span className="text-[10px] leading-3 text-muted-foreground">{t("viewer.slider.morphDisabled")}</span>}
      </div>
    </div>
  );
}
