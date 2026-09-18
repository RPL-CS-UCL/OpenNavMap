import anser from "anser";
import { ArrowDownToLine, Copy, Download, WrapText } from "lucide-react";
import { useMemo, useState } from "react";
import { Virtuoso } from "react-virtuoso";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Toggle } from "@/components/ui/toggle";
import { t } from "@/i18n";
import { cn } from "@/lib/utils";

interface Props {
  lines: string[];
  title?: string;
  onDownload?: () => void;
  className?: string;
  /** jsdom has no layout; tests pass this so Virtuoso renders rows. */
  initialItemCount?: number;
}

function AnsiLine({ text }: { text: string | undefined }) {
  // Virtuoso can call itemContent before `data` is wired on its initialItemCount render pass.
  const parts = useMemo(() => anser.ansiToJson(text ?? "", { use_classes: false, remove_empty: true }), [text]);
  return (
    <>
      {parts.map((p, i) => (
        <span key={i} style={p.fg ? { color: `rgb(${p.fg})` } : undefined}>
          {p.content}
        </span>
      ))}
    </>
  );
}

export function LogConsole({ lines, title, onDownload, className, initialItemCount }: Props) {
  const [follow, setFollow] = useState(true);
  const [wrap, setWrap] = useState(false);
  const [query, setQuery] = useState("");
  const shown = useMemo(() => {
    if (!query) return lines;
    const q = query.toLowerCase();
    return lines.filter((l) => l.toLowerCase().includes(q));
  }, [lines, query]);

  const copy = () => {
    navigator.clipboard?.writeText(shown.join("\n")).then(() => toast.success(t("log.copied")));
  };
  const download = () => {
    if (onDownload) return onDownload();
    const blob = new Blob([lines.join("\n")], { type: "text/plain" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${title ?? "job"}.log`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  return (
    <div className={cn("flex h-full min-h-0 flex-col border bg-[#111] text-[#ddd]", className)}>
      <div className="flex items-center gap-1 border-b border-white/10 px-2 py-1 text-[11px]">
        {title && <span className="mr-2 font-mono text-white/70">{title}</span>}
        <Toggle size="sm" pressed={follow} onPressedChange={setFollow} aria-label={t("log.follow")} className="h-6 px-1.5">
          <ArrowDownToLine className="h-3 w-3" />
        </Toggle>
        <Toggle size="sm" pressed={wrap} onPressedChange={setWrap} aria-label={t("log.wrap")} className="h-6 px-1.5">
          <WrapText className="h-3 w-3" />
        </Toggle>
        <Input
          type="search"
          role="searchbox"
          aria-label={t("log.search")}
          placeholder={t("log.search")}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="ml-2 h-6 w-48 border-white/20 bg-transparent text-[11px] text-white"
        />
        <span className="ml-auto font-mono tabular-nums text-white/50">
          {shown.length} / {lines.length}
        </span>
        <Button variant="ghost" size="icon" className="h-6 w-6 text-white/70" aria-label={t("log.copy")} onClick={copy}>
          <Copy className="h-3 w-3" />
        </Button>
        <Button variant="ghost" size="icon" className="h-6 w-6 text-white/70" aria-label={t("log.download")} onClick={download}>
          <Download className="h-3 w-3" />
        </Button>
      </div>
      {shown.length === 0 ? (
        <div className="p-2 font-mono text-[11px] text-white/40">{t("log.empty")}</div>
      ) : (
        <Virtuoso
          className="min-h-0 flex-1"
          data={shown}
          initialItemCount={initialItemCount}
          followOutput={follow ? "smooth" : false}
          itemContent={(_, line) => (
            <div
              className={cn(
                "px-2 font-mono text-[11.5px] leading-[18px]",
                wrap ? "whitespace-pre-wrap break-all" : "whitespace-pre",
              )}
            >
              <AnsiLine text={line} />
            </div>
          )}
        />
      )}
    </div>
  );
}
