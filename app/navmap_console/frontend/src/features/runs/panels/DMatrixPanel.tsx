import { useEffect, useRef, useState } from "react";
import type { MouseEvent as ReactMouseEvent, PointerEvent as ReactPointerEvent, WheelEvent as ReactWheelEvent } from "react";
import { dmatrixPngUrl, useDMatrix } from "@/api/hooks/use-results";
import type { Scene } from "@/api/scene-bundle";
import type { DMatrix } from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { t } from "@/i18n";
import { useSceneStore } from "@/stores/scene-store";
import { PairCard, type PairMeta } from "./PairCard";
import { MATRIX_CELL_PX, cellAtPoint, findLoopIndex, nearestCandidate, zoomAtPoint, type PanZoom } from "./dmatrix-math";

function MatrixView({ rid, runId, step, scene, d }: {
  rid: string; runId: string; step: number; scene: Scene | null; d: DMatrix;
}) {
  const [view, setView] = useState<PanZoom>({ zoom: 1, pan: { x: 0, y: 0 } });
  const [pair, setPair] = useState<{ a: number; b: number; meta?: PairMeta } | null>(null);
  const [candidateIdx, setCandidateIdx] = useState(0);
  const drag = useRef<{ start: { x: number; y: number }; pan: { x: number; y: number } } | null>(null);

  const cols = d.col_node_ids.length;
  const rows = d.row_node_ids.length;
  const rowOf = new Map(d.row_node_ids.map((n, i) => [n, i] as const));
  const colOf = new Map(d.col_node_ids.map((n, i) => [n, i] as const));

  // Arrow keys cycle through candidates and mirror the selection in the 3D scene.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA")) return;
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      if (d.candidates.length === 0) return;
      const delta = e.key === "ArrowRight" ? 1 : -1;
      const idx = (candidateIdx + delta + d.candidates.length) % d.candidates.length;
      setCandidateIdx(idx);
      const c = d.candidates[idx];
      const li = findLoopIndex(scene, c.db, c.query);
      if (li >= 0) useSceneStore.getState().select({ kind: "loop", index: li });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [d, scene, candidateIdx]);

  const onOverlayClick = (e: ReactMouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const { col, row } = cellAtPoint(e.clientX - rect.left, e.clientY - rect.top, view.zoom, view.pan);
    const c = nearestCandidate(col, row, d.candidates, rowOf, colOf, 1.5);
    if (!c) return;
    setCandidateIdx(d.candidates.indexOf(c));
    const f = d.factors.find((x) => x.db === c.db && x.query === c.query);
    setPair({ a: c.db, b: c.query, meta: f ? { weight: f.weight, conf: f.conf } : undefined });
    const li = findLoopIndex(scene, c.db, c.query);
    if (li >= 0) useSceneStore.getState().select({ kind: "loop", index: li });
  };

  const onWheel = (e: ReactWheelEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const factor = e.deltaY < 0 ? 1.25 : 0.8;
    setView((v) => zoomAtPoint(v.zoom, v.pan, { x: e.clientX - rect.left, y: e.clientY - rect.top }, factor));
  };

  const onPointerDown = (e: ReactPointerEvent<SVGSVGElement>) => {
    drag.current = { start: { x: e.clientX, y: e.clientY }, pan: view.pan };
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: ReactPointerEvent<SVGSVGElement>) => {
    if (!drag.current) return;
    const d0 = drag.current;
    setView((v) => ({ ...v, pan: { x: d0.pan.x + e.clientX - d0.start.x, y: d0.pan.y + e.clientY - d0.start.y } }));
  };
  const onPointerUp = () => {
    drag.current = null;
  };

  return (
    <div className="relative h-full w-full overflow-hidden">
      <div style={{ transform: `translate(${view.pan.x}px, ${view.pan.y}px) scale(${view.zoom})`, transformOrigin: "0 0" }}>
        <div className="relative" style={{ width: cols * MATRIX_CELL_PX, height: rows * MATRIX_CELL_PX }}>
          <img
            src={dmatrixPngUrl(rid, runId, step)}
            alt="D-matrix"
            width={cols * MATRIX_CELL_PX}
            height={rows * MATRIX_CELL_PX}
            className="block"
            style={{ imageRendering: "pixelated" }}
            draggable={false}
          />
          <svg
            data-testid="matrix-overlay"
            viewBox={`0 0 ${cols} ${rows}`}
            width={cols * MATRIX_CELL_PX}
            height={rows * MATRIX_CELL_PX}
            className="absolute inset-0 cursor-crosshair"
            onClick={onOverlayClick}
            onWheel={onWheel}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerLeave={onPointerUp}
          >
            {d.candidates.map((c, i) => {
              const cc = colOf.get(c.query);
              const rr = rowOf.get(c.db);
              if (cc === undefined || rr === undefined) return null;
              return (
                <rect key={`c${i}`} data-testid="candidate-marker" x={cc + 0.15} y={rr + 0.15} width={0.7} height={0.7}
                  fill="none" stroke={c.stage === "gv" ? "#4f83cc" : "#9ca3af"} strokeWidth={0.08} />
              );
            })}
            {d.factors.map((f, i) => {
              const cc = colOf.get(f.query);
              const rr = rowOf.get(f.db);
              if (cc === undefined || rr === undefined) return null;
              const cx = cc + 0.5;
              const cy = rr + 0.5;
              return f.accepted ? (
                <circle key={`f${i}`} data-testid={`factor-marker-${f.db}-${f.query}`} cx={cx} cy={cy} r={0.28} fill="#16a34a" />
              ) : (
                <g key={`f${i}`} data-testid={`factor-marker-${f.db}-${f.query}`} stroke="#dc2626" strokeWidth={0.08}>
                  <line x1={cx - 0.2} y1={cy - 0.2} x2={cx + 0.2} y2={cy + 0.2} />
                  <line x1={cx - 0.2} y1={cy + 0.2} x2={cx + 0.2} y2={cy - 0.2} />
                </g>
              );
            })}
          </svg>
        </div>
      </div>
      <div className="absolute right-2 top-2 z-10 flex flex-col gap-0.5 rounded-sm bg-background/80 p-1 text-[10px] text-muted-foreground">
        <span className="flex items-center gap-1"><span className="inline-block h-2 w-2 border border-[#9ca3af]" />{t("panel.dmatrix.legend.vpr")}</span>
        <span className="flex items-center gap-1"><span className="inline-block h-2 w-2 border border-[#4f83cc]" />{t("panel.dmatrix.legend.gv")}</span>
        <span className="flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-full bg-[#16a34a]" />{t("panel.dmatrix.legend.accepted")}</span>
        <span className="flex items-center gap-1 text-[#dc2626]">✕{t("panel.dmatrix.legend.rejected")}</span>
      </div>
      <PairCard rid={rid} runId={runId} a={pair?.a ?? 0} b={pair?.b ?? 0} meta={pair?.meta}
        open={pair !== null} onOpenChange={(o) => !o && setPair(null)} />
    </div>
  );
}

/** VPR distance-matrix image with pan/zoom, candidate overlay and keyboard cycling. */
export function DMatrixPanel({ rid, runId, step, scene }: {
  rid: string; runId: string; step: number; scene: Scene | null;
}) {
  const dm = useDMatrix(rid, runId, step);
  if (dm.isPending) return <Skeleton className="m-2 h-full" />;
  if (dm.isError) return <div className="p-2 text-xs text-destructive">{t("panel.dmatrix.noData")}</div>;
  if (dm.data === null) {
    // Legacy results: only the matplotlib jpg exists, no cell data to overlay.
    return (
      <div className="relative flex h-full items-center justify-center overflow-hidden">
        <Badge variant="secondary" className="absolute right-2 top-2 z-10">{t("panel.dmatrix.legacy")}</Badge>
        <img src={dmatrixPngUrl(rid, runId, step)} alt="D-matrix" className="max-h-full max-w-full object-contain" />
      </div>
    );
  }
  return <MatrixView rid={rid} runId={runId} step={step} scene={scene} d={dm.data} />;
}
