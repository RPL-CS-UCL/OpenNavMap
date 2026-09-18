import { ChevronDown, ChevronUp } from "lucide-react";
import { useJobs } from "@/api/hooks/use-jobs";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { JobStatusBadge, progressText } from "@/features/jobs/job-status";
import { LogConsole } from "@/features/jobs/LogConsole";
import { useJobLog } from "@/features/jobs/use-job-log";
import { t } from "@/i18n";
import { useHotkey } from "@/lib/hotkeys";
import { cn } from "@/lib/utils";
import { useJobDockStore } from "@/stores/job-dock-store";

export function JobDock() {
  const { open, activeJobId, toggle, setActive } = useJobDockStore();
  const jobs = useJobs("");
  useHotkey("j", toggle, { ctrl: true });
  const list = jobs.data ?? [];
  const running = list.find((j) => j.status === "running") ?? list.find((j) => j.status === "queued");
  const active = list.find((j) => j.id === activeJobId) ?? running ?? null;
  const { lines } = useJobLog(open && active ? active.id : null);
  const pct = active?.progress.total ? (active.progress.completed_steps / active.progress.total) * 100 : null;

  return (
    <div className={cn("col-start-2 border-t bg-background", open ? "h-72" : "h-8")}>
      <div className="flex h-8 items-center gap-2 px-2 text-xs">
        <Button variant="ghost" size="icon" className="h-6 w-6" aria-label={t("dock.toggle")} onClick={toggle}>
          {open ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronUp className="h-3.5 w-3.5" />}
        </Button>
        {active ? (
          <>
            <JobStatusBadge status={active.status} />
            <span className="font-mono">{active.id}</span>
            <span className="tabular-nums">{progressText(active.progress)}</span>
            {active.progress.stage && <span className="text-muted-foreground">{active.progress.stage}</span>}
            {pct !== null && <Progress value={pct} className="h-1.5 w-40" />}
          </>
        ) : (
          <span className="text-muted-foreground">{t("dock.none")}</span>
        )}
      </div>
      {open && (
        <div className="grid h-[calc(100%-2rem)] grid-cols-[220px_1fr] border-t">
          <ul className="overflow-auto text-xs">
            {list.slice(0, 50).map((j) => (
              <li key={j.id}>
                <button
                  type="button"
                  className={cn(
                    "flex w-full items-center gap-2 px-2 py-1 text-left hover:bg-accent",
                    active?.id === j.id && "bg-accent",
                  )}
                  onClick={() => setActive(j.id)}
                >
                  <JobStatusBadge status={j.status} />
                  <span className="truncate font-mono">{j.id}</span>
                </button>
              </li>
            ))}
          </ul>
          {active ? (
            <LogConsole lines={lines} title={active.id} />
          ) : (
            <div className="p-2 text-xs text-muted-foreground">{t("dock.pick")}</div>
          )}
        </div>
      )}
    </div>
  );
}
