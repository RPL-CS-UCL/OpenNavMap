import { ArrowUp, File, Folder } from "lucide-react";
import { useEffect, useState } from "react";
import { useFsList, useFsRoots } from "@/api/hooks/use-fs";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { t } from "@/i18n";
import { formatBytes } from "@/lib/format";
import { cn } from "@/lib/utils";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onChoose: (path: string) => void;
}

function withinRoots(path: string, roots: string[]): boolean {
  return roots.some((r) => path === r || path.startsWith(r.endsWith("/") ? r : `${r}/`));
}

export function FsBrowserDialog({ open, onOpenChange, onChoose }: Props) {
  // current === null shows the list of allowed roots.
  const [current, setCurrent] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const roots = useFsRoots();
  const listing = useFsList(current);

  useEffect(() => {
    if (open && current === null && roots.data && roots.data.roots.length === 1) {
      setCurrent(roots.data.roots[0]);
    }
  }, [open, current, roots.data]);

  const enter = (path: string) => {
    setCurrent(path);
    setSelected(null);
  };
  const up = () => {
    if (listing.data?.parent) enter(listing.data.parent);
    else enter(null as unknown as string);
  };

  const crumbs = (() => {
    if (!current) return [];
    const parts = current.split("/").filter(Boolean);
    return parts.map((seg, i) => ({ seg, path: `/${parts.slice(0, i + 1).join("/")}` }));
  })();
  const rootList = roots.data?.roots ?? [];
  const chosen = selected ?? current;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>{t("fs.title")}</DialogTitle>
        </DialogHeader>
        <div className="flex items-center gap-1 overflow-x-auto font-mono text-xs">
          <Button variant="ghost" size="sm" className="h-7 px-2" onClick={up} disabled={current === null}>
            <ArrowUp className="mr-1 h-3 w-3" />
            {t("fs.up")}
          </Button>
          <span className="text-muted-foreground">/</span>
          {crumbs.map((c) => (
            <span key={c.path} className="flex items-center gap-1">
              <button
                type="button"
                className={cn("hover:underline", !withinRoots(c.path, rootList) && "pointer-events-none text-muted-foreground")}
                onClick={() => enter(c.path)}
              >
                {c.seg}
              </button>
              <span className="text-muted-foreground">/</span>
            </span>
          ))}
        </div>
        <ScrollArea className="h-80 rounded-sm border">
          <ul className="divide-y text-sm">
            {current === null &&
              rootList.map((r) => (
                <li key={r}>
                  <button type="button" className="flex w-full items-center gap-2 px-3 py-1.5 text-left hover:bg-accent" onDoubleClick={() => enter(r)} onClick={() => enter(r)}>
                    <Folder className="h-4 w-4 text-muted-foreground" />
                    <span className="font-mono text-xs">{r}</span>
                  </button>
                </li>
              ))}
            {current !== null && listing.isPending && <li className="px-3 py-2 text-muted-foreground">{t("common.loading")}</li>}
            {current !== null && listing.error && (
              <li className="px-3 py-2 text-destructive">{t("common.error", { detail: listing.error.message })}</li>
            )}
            {listing.data && listing.data.entries.length === 0 && (
              <li className="px-3 py-2 text-muted-foreground">{t("fs.empty")}</li>
            )}
            {listing.data?.entries.map((e) => (
              <li key={e.path}>
                <button
                  type="button"
                  disabled={!e.is_dir}
                  className={cn(
                    "flex w-full items-center gap-2 px-3 py-1.5 text-left hover:bg-accent disabled:opacity-50",
                    selected === e.path && "bg-accent",
                  )}
                  onClick={() => setSelected(e.path)}
                  onDoubleClick={() => e.is_dir && enter(e.path)}
                >
                  {e.is_dir ? <Folder className="h-4 w-4 text-muted-foreground" /> : <File className="h-4 w-4 text-muted-foreground" />}
                  <span className="flex-1 font-mono text-xs">{e.name}</span>
                  {e.looks_like_session && <Badge variant="secondary">{t("fs.sessionLike")}</Badge>}
                  {!e.is_dir && e.size !== null && <span className="text-xs text-muted-foreground">{formatBytes(e.size)}</span>}
                </button>
              </li>
            ))}
          </ul>
        </ScrollArea>
        <DialogFooter className="items-center">
          <span className="mr-auto truncate font-mono text-xs text-muted-foreground" title={chosen ?? ""}>
            {chosen ?? ""}
          </span>
          <Button variant="outline" size="sm" onClick={() => onOpenChange(false)}>
            {t("common.cancel")}
          </Button>
          <Button size="sm" disabled={!chosen} onClick={() => chosen && onChoose(chosen)}>
            {t("fs.choose")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
