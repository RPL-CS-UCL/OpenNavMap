import type { CullRow } from "@/api/types";
import { useCulling } from "@/api/hooks/use-results";
import { EmptyState } from "@/components/common/EmptyState";
import { Badge } from "@/components/ui/badge";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { t, type MessageKey } from "@/i18n";
import { useSceneStore } from "@/stores/scene-store";

const METHOD_KEYS: Record<string, MessageKey> = {
  culled_by_iqa: "panel.culling.method.culled_by_iqa",
  culled_by_forward: "panel.culling.method.culled_by_forward",
  culled_by_backward: "panel.culling.method.culled_by_backward",
};

const badgeVariant = (method: string) => {
  if (method.includes("iqa")) return "secondary" as const;
  if (method.includes("forward") || method.includes("backward")) return "outline" as const;
  return "default" as const;
};

function Row({ row, onSelect }: { row: CullRow; onSelect: () => void }) {
  const prob = row.prob ?? 0;
  return (
    <button
      type="button"
      data-testid={`cull-row-${row.node_id}`}
      onClick={onSelect}
      className="flex w-full items-center gap-1.5 rounded-sm border p-1 text-left hover:bg-muted/60"
    >
      <img
        src={row.image_url}
        alt={`node ${row.node_id}`}
        className="h-10 w-14 shrink-0 rounded-sm border object-cover"
      />
      {row.other_image_url && (
        <img
          src={row.other_image_url}
          alt={`vs ${row.other ?? "?"}`}
          className="h-10 w-14 shrink-0 rounded-sm border object-cover"
        />
      )}
      <Badge variant={badgeVariant(row.method)} className="shrink-0">
        {METHOD_KEYS[row.method] ? t(METHOD_KEYS[row.method]) : row.method}
      </Badge>
      <span className="ml-auto flex w-16 shrink-0 flex-col gap-0.5">
        <span className="self-end font-mono tabular-nums">{row.prob === null ? "–" : row.prob.toFixed(2)}</span>
        <span className="h-1 w-full overflow-hidden rounded-full bg-muted">
          <span className="block h-full bg-primary/70" style={{ width: `${Math.round(prob * 100)}%` }} />
        </span>
      </span>
    </button>
  );
}

/** Per-step culling outcomes: culled rows with image pairs, kept rows in a collapsed section. */
export function CullingPanel({ rid, runId, step }: { rid: string; runId: string; step: number }) {
  const { data, isPending, isError } = useCulling(rid, runId, step);
  if (isPending) return <div className="p-2 text-muted-foreground">…</div>;
  if (isError || !data) {
    return <EmptyState title={t("common.error", { detail: "culling.json" })} />;
  }
  const select = (nodeId: number, culled: boolean) => {
    const s = useSceneStore.getState();
    s.select({ kind: "node", id: nodeId });
    s.setLayer("culled", culled);
  };
  if (data.culled.length === 0 && data.kept.length === 0) {
    return <EmptyState title={t("panel.culling.empty")} />;
  }
  return (
    <div className="flex h-full min-h-0 flex-col gap-1.5 overflow-auto p-2">
      <span className="text-muted-foreground">{t("panel.culling.culled")}</span>
      {data.culled.length === 0 ? (
        <div className="text-muted-foreground">{t("panel.culling.empty")}</div>
      ) : (
        data.culled.map((row) => <Row key={row.node_id} row={row} onSelect={() => select(row.node_id, true)} />)
      )}
      <Collapsible className="flex flex-col gap-1">
        <CollapsibleTrigger className="flex items-center gap-1 text-muted-foreground hover:text-foreground">
          {t("panel.culling.kept")} <span className="font-mono">({data.kept.length})</span>
        </CollapsibleTrigger>
        <CollapsibleContent className="flex flex-col gap-1.5">
          {data.kept.map((row) => <Row key={row.node_id} row={row} onSelect={() => select(row.node_id, false)} />)}
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}
