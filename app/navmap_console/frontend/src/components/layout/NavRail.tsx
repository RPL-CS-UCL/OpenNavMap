import { ListChecks, Map, Settings } from "lucide-react";
import { NavLink } from "react-router-dom";
import { useJobs } from "@/api/hooks/use-jobs";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { t } from "@/i18n";
import { cn } from "@/lib/utils";

const items = [
  { to: "/regions", label: t("nav.regions"), icon: Map },
  { to: "/jobs", label: t("nav.jobs"), icon: ListChecks },
  { to: "/settings", label: t("nav.settings"), icon: Settings },
];

export function NavRail() {
  const active = useJobs("queued,running");
  const n = active.data?.length ?? 0;
  return (
    <nav className="row-span-3 flex flex-col items-center gap-1 border-r bg-muted/40 py-2" aria-label="Primary">
      <div className="mb-2 flex h-8 w-8 items-center justify-center font-mono text-xs font-semibold" title={t("app.title")}>
        NM
      </div>
      {items.map(({ to, label, icon: Icon }) => (
        <Tooltip key={to}>
          <TooltipTrigger asChild>
            <NavLink
              to={to}
              aria-label={label}
              className={({ isActive }) =>
                cn(
                  "relative flex h-9 w-9 items-center justify-center rounded-sm text-muted-foreground hover:bg-accent hover:text-foreground",
                  isActive && "bg-accent text-foreground",
                )
              }
            >
              <Icon className="h-4 w-4" />
              {to === "/jobs" && n > 0 && (
                <span className="absolute -right-0.5 -top-0.5 rounded-full bg-primary px-1 font-mono text-[9px] leading-3 text-primary-foreground">
                  {n}
                </span>
              )}
            </NavLink>
          </TooltipTrigger>
          <TooltipContent side="right">{label}</TooltipContent>
        </Tooltip>
      ))}
    </nav>
  );
}
