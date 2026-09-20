import { useEffect, useMemo } from "react";
import { LOOP_ACCEPTED, LOOP_HIST, LOOP_OVERTURNED, LOOP_REJECTED_NEW, NODE_CULLED, NODE_NEW, NODE_NOT_COVIS, type Scene } from "@/api/scene-bundle";
import { useNodeDetail, useStepSummaries, nodeImageUrl } from "@/api/hooks/use-results";
import type { NodeDetail, StepSummary } from "@/api/types";
import { EmptyState } from "@/components/common/EmptyState";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { t } from "@/i18n";
import { midpoints } from "@/scene/build/segments";
import { nodeRowIndex } from "@/scene/build/visibility";
import { useSceneStore } from "@/stores/scene-store";

const fmt = (v: number, digits = 2) => (Number.isNaN(v) ? "–" : v.toFixed(digits));

function SummaryBlock({ s }: { s: StepSummary | undefined }) {
  if (!s) return <EmptyState title={t("viewer.inspector.none")} />;
  const rows: [string, string][] = [
    [t("viewer.inspector.session"), s.session_id],
    [t("viewer.inspector.directory"), s.dir_name],
    [t("viewer.inspector.nodes"), `${s.num_nodes} (${t("viewer.inspector.newCount", { count: s.num_new })} / ${t("viewer.inspector.culledCount", { count: s.num_culled })})`],
    [t("viewer.inspector.components"), s.component_sizes.join(", ")],
    [t("viewer.inspector.edges"), `o${s.edge_counts.odom} c${s.edge_counts.covis} t${s.edge_counts.trav}`],
    [t("viewer.inspector.loops"),
      `${t("viewer.inspector.loopsAccepted", { count: s.loops.accepted })} / ${t("viewer.inspector.loopsRejected", { count: s.loops.rejected_new })} / ${t("viewer.inspector.loopsOverturned", { count: s.loops.overturned_hist })}`],
    [t("viewer.inspector.pgo"), s.pgo_error_final === null ? "–" : `${s.pgo_error_initial?.toFixed(2) ?? "–"} → ${s.pgo_error_final.toFixed(2)}`],
    [t("viewer.inspector.moved"), `${s.num_moved} (${s.max_displacement.toFixed(3)} m)`],
    [t("viewer.inspector.duration"), s.duration_s === null ? "–" : `${s.duration_s.toFixed(1)} s`],
  ];
  return (
    <div className="flex flex-col gap-2 p-2 text-xs">
      <div className="font-mono text-[11px] text-muted-foreground">{s.dir_name}</div>
      {rows.map(([k, v]) => (
        <div key={k} className="flex justify-between gap-2">
          <span className="text-muted-foreground">{k}</span>
          <span className="text-right font-mono">{v}</span>
        </div>
      ))}
      {s.precision.length > 0 && (
        <div className="flex flex-wrap gap-1">
          <Badge variant="secondary">P@{s.precision.length} {s.precision.map((p) => p.toFixed(2)).join(", ")}</Badge>
          <Badge variant="secondary">R@{s.recall.length} {s.recall.map((p) => p.toFixed(2)).join(", ")}</Badge>
        </div>
      )}
    </div>
  );
}

function NodeBlock({ rid, runId, detail, scene, row, requestFly, togglePin }: {
  rid: string; runId: string; detail: NodeDetail | undefined; scene: Scene; row: number;
  requestFly: (p: [number, number, number]) => void; togglePin: (k: number) => void;
}) {
  if (!detail) return <Skeleton className="m-2 h-24" />;
  const k = scene.step[row];
  const flagNames: string[] = [];
  if (detail.flags & NODE_NEW) flagNames.push("new");
  if (detail.flags & NODE_CULLED) flagNames.push("culled");
  if (detail.flags & NODE_NOT_COVIS) flagNames.push("not-covis");
  const rows: [string, string][] = [
    [t("viewer.inspector.session"), detail.session_id],
    [t("viewer.inspector.frame"), detail.frame],
    [t("viewer.inspector.degree"), `o${detail.degree.odom} c${detail.degree.covis} t${detail.degree.trav}`],
    [t("viewer.inspector.position"), `[${detail.pos.map((v) => v.toFixed(2)).join(", ")}]`],
    [t("viewer.inspector.flags"), flagNames.join(", ") || "–"],
  ];
  if (detail.cull) rows.push([t("viewer.inspector.cull"), `${detail.cull.method} p=${detail.cull.prob === null ? "–" : detail.cull.prob.toFixed(2)}`]);
  return (
    <div className="flex flex-col gap-2 p-2 text-xs">
      <img src={nodeImageUrl(rid, runId, detail.node_id, 512)} alt="" className="aspect-[16/9] w-full rounded-sm bg-muted object-cover" />
      <div className="font-mono text-[11px] text-muted-foreground">
        node {detail.node_id} · step {k} · {detail.frame}
      </div>
      {rows.map(([key, v]) => (
        <div key={key} className="flex justify-between gap-2">
          <span className="text-muted-foreground">{key}</span>
          <span className="text-right font-mono">{v}</span>
        </div>
      ))}
      {detail.loops.length > 0 && (
        <div className="flex flex-col gap-0.5">
          <span className="text-muted-foreground">{t("viewer.inspector.loops")}</span>
          {detail.loops.map((l, i) => (
            <span key={i} className="font-mono text-[11px]">
              ↔ {l.other} w={fmt(l.weight)} {l.accepted ? "✓" : "✗"}
            </span>
          ))}
        </div>
      )}
      <div className="flex gap-1">
        <Button size="sm" variant="outline" onClick={() => requestFly([scene.pos[row * 3], scene.pos[row * 3 + 1], scene.pos[row * 3 + 2]])}>
          {t("viewer.inspector.flyTo")}
        </Button>
        <Button size="sm" variant="outline" onClick={() => togglePin(k)}>
          {t("viewer.inspector.pinStep", { step: k })}
        </Button>
      </div>
    </div>
  );
}

function LoopBlock({ scene, index, requestFly }: { scene: Scene; index: number; requestFly: (p: [number, number, number]) => void }) {
  const db = scene.loopIdx[index * 2];
  const query = scene.loopIdx[index * 2 + 1];
  const f = scene.loopFlags[index];
  const badges: string[] = [];
  if (f & LOOP_ACCEPTED) badges.push(t("viewer.loop.accepted"));
  if (f & LOOP_HIST) badges.push(t("viewer.loop.hist"));
  if (f & LOOP_OVERTURNED) badges.push(t("viewer.loop.overturned"));
  if (f & LOOP_REJECTED_NEW) badges.push(t("viewer.loop.rejectedNew"));
  const mid = midpoints(scene.pos, scene.loopIdx.subarray(index * 2, index * 2 + 2));
  return (
    <div className="flex flex-col gap-2 p-2 text-xs">
      <div className="flex flex-wrap gap-1">
        {badges.map((b) => <Badge key={b} variant="secondary">{b}</Badge>)}
      </div>
      <div className="flex justify-between gap-2">
        <span className="text-muted-foreground">db</span>
        <span className="font-mono">{db}</span>
      </div>
      <div className="flex justify-between gap-2">
        <span className="text-muted-foreground">query</span>
        <span className="font-mono">{query}</span>
      </div>
      <div className="flex justify-between gap-2">
        <span className="text-muted-foreground">conf</span>
        <span className="font-mono">{fmt(scene.loopConf[index])}</span>
      </div>
      <div className="flex justify-between gap-2">
        <span className="text-muted-foreground">weight</span>
        <span className="font-mono">{fmt(scene.loopWeight[index])}</span>
      </div>
      <div className="flex justify-between gap-2">
        <span className="text-muted-foreground">t_err / r_err</span>
        <span className="font-mono">{fmt(scene.loopTerr[index])} / {fmt(scene.loopRerr[index])}</span>
      </div>
      <Button size="sm" variant="outline" onClick={() => requestFly([mid[0], mid[1], mid[2]])}>
        {t("viewer.inspector.flyTo")}
      </Button>
    </div>
  );
}

export function Inspector({ rid, runId, scene }: { rid: string; runId: string; scene: Scene | null }) {
  const selected = useSceneStore((s) => s.selected);
  const step = useSceneStore((s) => s.step);
  const requestFly = useSceneStore((s) => s.requestFly);
  const togglePin = useSceneStore((s) => s.togglePin);
  const summaries = useStepSummaries(rid, runId);
  const rowOf = useMemo(() => (scene ? nodeRowIndex(scene.nodeId) : new Map<number, number>()), [scene]);

  const nodeSel = selected?.kind === "node" ? selected : null;
  const nodeRow = nodeSel ? rowOf.get(nodeSel.id) : undefined;
  const nodeStep = nodeRow !== undefined && scene ? scene.step[nodeRow] : null;
  const detail = useNodeDetail(rid, runId, nodeStep, nodeSel ? nodeSel.id : null);

  // Fly the camera to whatever just got selected.
  useEffect(() => {
    if (!scene || !selected) return;
    if (selected.kind === "node") {
      const r = rowOf.get(selected.id);
      if (r === undefined) return;
      requestFly([scene.pos[r * 3], scene.pos[r * 3 + 1], scene.pos[r * 3 + 2]]);
    } else if (selected.index < scene.numLoops) {
      const m = midpoints(scene.pos, scene.loopIdx.subarray(selected.index * 2, selected.index * 2 + 2));
      requestFly([m[0], m[1], m[2]]);
    }
  }, [selected, scene, rowOf, requestFly]);

  if (!selected) {
    if (!scene) return <EmptyState title={t("viewer.inspector.none")} body={t("viewer.inspector.noScene")} />;
    return <SummaryBlock s={summaries.data?.[step ?? -1]} />;
  }
  if (selected.kind === "node") {
    if (!scene || nodeRow === undefined) return <EmptyState title={t("viewer.inspector.node")} />;
    return <NodeBlock rid={rid} runId={runId} detail={detail.data} scene={scene} row={nodeRow} requestFly={requestFly} togglePin={togglePin} />;
  }
  if (!scene || selected.index >= scene.numLoops) return <EmptyState title={t("viewer.inspector.loop")} />;
  return <LoopBlock scene={scene} index={selected.index} requestFly={requestFly} />;
}
