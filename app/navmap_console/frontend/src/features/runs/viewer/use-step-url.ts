import { useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { useSceneStore } from "@/stores/scene-store";

export interface ViewerEdge { kind: "loop"; index: number }
export interface ViewerParams { step?: number; node?: number; edge?: ViewerEdge }

const uint = (v: string | null): number | undefined => {
  if (v === null || !/^\d+$/.test(v)) return undefined;
  return Number(v);
};

export function parseViewerParams(sp: URLSearchParams): ViewerParams {
  const out: ViewerParams = {};
  const step = uint(sp.get("step"));
  if (step !== undefined) out.step = step;
  const node = uint(sp.get("node"));
  if (node !== undefined) out.node = node;
  const m = sp.get("edge")?.match(/^loop:(\d+)$/);
  if (m) out.edge = { kind: "loop", index: Number(m[1]) };
  return out;
}

export function paramsToSearch(p: ViewerParams): string {
  const sp = new URLSearchParams();
  if (p.step !== undefined) sp.set("step", String(p.step));
  if (p.node !== undefined) sp.set("node", String(p.node));
  if (p.edge) sp.set("edge", `loop:${p.edge.index}`);
  return sp.toString();
}

/** The store is the source of truth; the URL is a mirror. Apply deep links on mount, then write store changes back. */
export function useStepUrl(): void {
  const [sp, setSp] = useSearchParams();

  useEffect(() => {
    const p = parseViewerParams(sp);
    if (p.step !== undefined) {
      const s = useSceneStore.getState();
      // setStep clamps against maxStep, which is still -1 before summaries arrive; lift it so the deep link survives.
      if (s.maxStep < p.step) s.setMaxStep(p.step);
      s.setStep(p.step);
    }
    if (p.node !== undefined) useSceneStore.getState().select({ kind: "node", id: p.node });
    else if (p.edge) useSceneStore.getState().select({ kind: "loop", index: p.edge.index });
    // oxlint-disable-next-line react/exhaustive-deps
  }, []);

  useEffect(() => {
    const unsubStep = useSceneStore.subscribe(
      (s) => s.step,
      (step) =>
        setSp(
          (prev) => {
            const next = new URLSearchParams(prev);
            const want = step === null ? null : String(step);
            if (next.get("step") === want) return prev;
            if (want === null) next.delete("step");
            else next.set("step", want);
            return next;
          },
          { replace: true },
        ),
    );
    const unsubSel = useSceneStore.subscribe(
      (s) => s.selected,
      (sel) =>
        setSp(
          (prev) => {
            const next = new URLSearchParams(prev);
            const wantNode = sel?.kind === "node" ? String(sel.id) : null;
            const wantEdge = sel?.kind === "loop" ? `loop:${sel.index}` : null;
            if (next.get("node") === wantNode && next.get("edge") === wantEdge) return prev;
            if (wantNode === null) next.delete("node");
            else next.set("node", wantNode);
            if (wantEdge === null) next.delete("edge");
            else next.set("edge", wantEdge);
            return next;
          },
          { replace: true },
        ),
    );
    return () => {
      unsubStep();
      unsubSel();
    };
  }, [setSp]);
}
