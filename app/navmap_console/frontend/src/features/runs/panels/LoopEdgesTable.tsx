import { flexRender } from "@tanstack/react-table";
import {
  getCoreRowModel,
  getSortedRowModel,
  legacyCreateColumnHelper,
  useLegacyTable,
  type LegacyColumnDef,
} from "@tanstack/react-table/legacy";
import { useEffect, useMemo, useState } from "react";
import { Virtuoso } from "react-virtuoso";
import { LOOP_ACCEPTED, LOOP_HIST, LOOP_OVERTURNED, type Scene } from "@/api/scene-bundle";
import { Badge } from "@/components/ui/badge";
import { EmptyState } from "@/components/common/EmptyState";
import { t } from "@/i18n";
import { cn } from "@/lib/utils";
import { midpoints } from "@/scene/build/segments";
import { useSceneStore } from "@/stores/scene-store";
import { PairCard, type PairMeta } from "./PairCard";

interface LoopRow {
  index: number;
  db: number;
  query: number;
  origin: "new" | "hist";
  conf: number;
  weight: number;
  accepted: boolean;
  overturned: boolean;
  terr: number | null;
  rerr: number | null;
}

const columnHelper = legacyCreateColumnHelper<LoopRow>();

function SortHeader({ label, onToggle }: { label: string; onToggle: () => void }) {
  return (
    <button type="button" className="hover:text-foreground" onClick={onToggle}>
      {label}
    </button>
  );
}

const GRID = "grid grid-cols-[2.5rem_2.5rem_3.5rem_3.5rem_5rem_6rem_3.5rem_3.5rem] gap-1";

export function LoopEdgesTable({ rid, runId, scene, initialItemCount }: {
  rid: string; runId: string; scene: Scene | null; initialItemCount?: number;
}) {
  const [sorting, setSorting] = useState<{ id: string; desc: boolean }[]>([]);
  const [pair, setPair] = useState<{ a: number; b: number; meta?: PairMeta } | null>(null);
  const [selectedLoop, setSelectedLoop] = useState<number | null>(null);

  const rows = useMemo<LoopRow[]>(() => {
    if (!scene) return [];
    const out: LoopRow[] = [];
    for (let i = 0; i < scene.numLoops; i++) {
      const f = scene.loopFlags[i];
      out.push({
        index: i,
        db: scene.loopIdx[i * 2],
        query: scene.loopIdx[i * 2 + 1],
        origin: f & LOOP_HIST ? "hist" : "new",
        conf: scene.loopConf[i],
        weight: scene.loopWeight[i],
        accepted: !!(f & LOOP_ACCEPTED),
        overturned: !!(f & LOOP_OVERTURNED),
        terr: Number.isNaN(scene.loopTerr[i]) ? null : scene.loopTerr[i],
        rerr: Number.isNaN(scene.loopRerr[i]) ? null : scene.loopRerr[i],
      });
    }
    return out;
  }, [scene]);

  const columns = useMemo((): LegacyColumnDef<LoopRow>[] => [
    columnHelper.display({ id: "db", header: ({ column }) => <SortHeader label={t("panel.loops.col.db")} onToggle={() => column.toggleSorting(column.getIsSorted() === "desc")} />, cell: (info) => <span className="font-mono">{info.row.original.db}</span>, sortFn: (a, b) => a.original.db - b.original.db }),
    columnHelper.display({ id: "query", header: ({ column }) => <SortHeader label={t("panel.loops.col.query")} onToggle={() => column.toggleSorting(column.getIsSorted() === "desc")} />, cell: (info) => <span className="font-mono">{info.row.original.query}</span>, sortFn: (a, b) => a.original.query - b.original.query }),
    columnHelper.display({ id: "origin", header: t("panel.loops.col.origin"), cell: (info) => {
      const r = info.row.original;
      return <Badge variant="secondary">{t(r.origin === "hist" ? "panel.loops.origin.hist" : "panel.loops.origin.new")}</Badge>;
    } }),
    columnHelper.display({ id: "conf", header: ({ column }) => <SortHeader label={t("panel.loops.col.conf")} onToggle={() => column.toggleSorting(column.getIsSorted() === "desc")} />, cell: (info) => <span data-testid="cell-conf" className="font-mono">{info.row.original.conf.toFixed(2)}</span>, sortFn: (a, b) => a.original.conf - b.original.conf }),
    columnHelper.display({ id: "weight", header: ({ column }) => <SortHeader label={t("panel.loops.col.weight")} onToggle={() => column.toggleSorting(column.getIsSorted() === "desc")} />, cell: (info) => {
      const w = info.row.original.weight;
      return (
        <span className="flex items-center gap-1">
          <span className="h-1 w-8 overflow-hidden rounded-sm bg-muted">
            <span className="block h-full bg-foreground" style={{ width: `${Math.round(w * 100)}%` }} />
          </span>
          <span className="font-mono">{w.toFixed(2)}</span>
        </span>
      );
    }, sortFn: (a, b) => a.original.weight - b.original.weight }),
    columnHelper.display({ id: "status", header: t("panel.loops.col.status"), cell: (info) => {
      const r = info.row.original;
      if (r.accepted) return <Badge>{t("panel.loops.status.accepted")}</Badge>;
      if (r.overturned) return <Badge variant="outline">{t("panel.loops.status.overturned")}</Badge>;
      return <Badge variant="destructive">{t("panel.loops.status.rejected")}</Badge>;
    } }),
    columnHelper.display({ id: "terr", header: t("panel.loops.col.terr"), cell: (info) => <span className="font-mono">{info.row.original.terr === null ? "–" : info.row.original.terr.toFixed(3)}</span> }),
    columnHelper.display({ id: "rerr", header: t("panel.loops.col.rerr"), cell: (info) => <span className="font-mono">{info.row.original.rerr === null ? "–" : info.row.original.rerr.toFixed(3)}</span> }),
  ], []);

  const table = useLegacyTable({
    data: rows,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  // Keep the highlighted row in sync with the 3D selection; scroll it into view.
  useEffect(() => {
    return useSceneStore.subscribe(
      (s) => (s.selected?.kind === "loop" ? s.selected.index : null),
      (idx) => {
        setSelectedLoop(idx);
        if (idx !== null) {
          document.querySelector(`[data-testid="loop-row-${idx}"]`)?.scrollIntoView({ block: "nearest" });
        }
      },
    );
  }, []);

  if (!scene || rows.length === 0) return <EmptyState title={t("panel.loops.empty")} />;

  const openPair = (r: LoopRow) => setPair({ a: r.db, b: r.query, meta: { weight: r.weight, conf: r.conf, terr: r.terr, rerr: r.rerr } });

  return (
    <div className="flex h-full min-h-0 flex-col text-xs">
      <div role="row" className={cn(GRID, "border-b px-1 py-0.5 font-medium text-muted-foreground")}>
        {table.getHeaderGroups()[0].headers.map((h) => (
          <div key={h.id}>{flexRender(h.column.columnDef.header, h.getContext())}</div>
        ))}
      </div>
      <Virtuoso
        className="min-h-0 flex-1"
        data={table.getRowModel().rows}
        initialItemCount={initialItemCount}
        itemContent={(_, row) => {
          const r = row.original;
          const mid = midpoints(scene.pos, scene.loopIdx.subarray(r.index * 2, r.index * 2 + 2));
          return (
            <div
              role="row"
              data-testid={`loop-row-${r.index}`}
              className={cn(GRID, "cursor-pointer border-b px-1 py-0.5 hover:bg-accent/60", selectedLoop === r.index && "bg-accent")}
              onMouseEnter={() => useSceneStore.getState().hover({ kind: "loop", index: r.index })}
              onMouseLeave={() => useSceneStore.getState().hover(null)}
              onClick={() => {
                useSceneStore.getState().select({ kind: "loop", index: r.index });
                useSceneStore.getState().requestFly([mid[0], mid[1], mid[2]]);
              }}
              onDoubleClick={() => openPair(r)}
            >
              {row.getVisibleCells().map((c) => (
                <div key={c.id}>{flexRender(c.column.columnDef.cell, c.getContext())}</div>
              ))}
            </div>
          );
        }}
      />
      <PairCard rid={rid} runId={runId} a={pair?.a ?? 0} b={pair?.b ?? 0} meta={pair?.meta}
        open={pair !== null} onOpenChange={(o) => !o && setPair(null)} />
    </div>
  );
}
