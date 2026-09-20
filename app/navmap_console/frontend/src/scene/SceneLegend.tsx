import { useMemo } from "react";
import { NODE_CULLED, NODE_NEW, type Scene } from "@/api/scene-bundle";
import { t } from "@/i18n";
import { useSceneStore } from "@/stores/scene-store";

interface Row { token: string; label: string }

function Swatch({ token }: { token: string }) {
  return <span className="inline-block h-2.5 w-2.5 shrink-0 rounded-sm" style={{ background: `hsl(var(--${token}))` }} />;
}

export function SceneLegend({ scene }: { scene: Scene | null }) {
  const mode = useSceneStore((s) => s.colorMode);
  const step = useSceneStore((s) => s.step) ?? 0;
  const pinned = useSceneStore((s) => s.pinned);
  const rows = useMemo<Row[]>(() => {
    const out: Row[] = [];
    if (mode === "focus") {
      out.push({ token: "scene-ref", label: t("scene.legend.reference") }, { token: "scene-current", label: t("scene.legend.current", { step }) });
      pinned.forEach((p, i) => out.push({ token: `scene-pin-${(i % 5) + 1}`, label: t("scene.legend.pinned", { step: p }) }));
    } else if (mode === "component" && scene) {
      const counts = new Map<number, number>();
      scene.comp.forEach((c) => counts.set(c, (counts.get(c) ?? 0) + 1));
      const sorted = [...counts.entries()].sort((a, b) => a[0] - b[0]);
      sorted.slice(0, 3).forEach(([c, n]) => out.push({ token: `series-${c + 1}`, label: t("scene.legend.component", { index: c, count: n }) }));
      const rest = sorted.slice(3);
      if (rest.length) out.push({ token: "scene-ghost", label: t("scene.legend.others", { count: rest.reduce((a, [, n]) => a + n, 0), groups: rest.length }) });
    } else if (mode === "new" && scene) {
      let fresh = 0;
      scene.flags.forEach((f) => { if (f & NODE_NEW) fresh++; });
      out.push({ token: "scene-current", label: t("scene.legend.new", { count: fresh }) }, { token: "scene-ref", label: t("scene.legend.referenceCount", { count: scene.numNodes - fresh }) });
    }
    if (scene && scene.flags.some((f) => (f & NODE_CULLED) !== 0)) out.push({ token: "scene-culled", label: t("scene.legend.culled") });
    out.push({ token: "scene-accept", label: t("scene.legend.accepted") }, { token: "scene-reject", label: t("scene.legend.rejected") });
    return out;
  }, [mode, step, pinned, scene]);

  return (
    <div className="pointer-events-none flex flex-col gap-1 rounded-sm border bg-background/85 px-2 py-1.5 text-[11px] leading-4 text-foreground backdrop-blur-sm">
      {mode === "order" && (
        <div className="flex items-center gap-1.5">
          <span>{t("scene.legend.orderStart")}</span>
          <span data-testid="legend-gradient" className="h-2 w-20 rounded-sm" style={{ background: "linear-gradient(to right, hsl(var(--seq-100)), hsl(var(--seq-700)))" }} />
          <span>{t("scene.legend.orderEnd", { step })}</span>
        </div>
      )}
      {rows.map((r) => (
        <div key={r.token + r.label} className="flex items-center gap-1.5"><Swatch token={r.token} /><span>{r.label}</span></div>
      ))}
    </div>
  );
}
