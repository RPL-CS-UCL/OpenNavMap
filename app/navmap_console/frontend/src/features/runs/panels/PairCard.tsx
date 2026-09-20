import { pairImageUrl } from "@/api/hooks/use-results";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { t } from "@/i18n";

export interface PairMeta {
  weight?: number;
  conf?: number;
  terr?: number | null;
  rerr?: number | null;
}

const num = (v: number | null | undefined, digits: number) =>
  v === undefined || v === null || Number.isNaN(v) ? "–" : v.toFixed(digits);

/** Side-by-side image of two nodes with the loop factor metrics underneath. */
export function PairCard({ rid, runId, a, b, meta, open, onOpenChange }: {
  rid: string; runId: string; a: number; b: number; meta?: PairMeta;
  open: boolean; onOpenChange: (o: boolean) => void;
}) {
  const rows: [string, string][] = [
    [t("panel.pair.weight"), num(meta?.weight, 2)],
    [t("panel.pair.conf"), num(meta?.conf, 2)],
    [t("panel.pair.terr"), num(meta?.terr, 3)],
    [t("panel.pair.rerr"), num(meta?.rerr, 3)],
  ];
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{t("panel.pair.title", { a, b })}</DialogTitle>
        </DialogHeader>
        <img src={pairImageUrl(rid, runId, a, b, 384)} alt={t("panel.pair.title", { a, b })} className="w-full rounded-sm bg-muted" />
        <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
          {rows.map(([k, v]) => (
            <div key={k} className="flex justify-between gap-2">
              <span className="text-muted-foreground">{k}</span>
              <span className="font-mono">{v}</span>
            </div>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  );
}
