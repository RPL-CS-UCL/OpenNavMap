import { ChevronRight, Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { Fragment } from "react";
import { Link, useMatches } from "react-router-dom";
import type { CrumbHandle } from "@/app/router";
import { Button } from "@/components/ui/button";
import { t } from "@/i18n";

function hasCrumb(handle: unknown): handle is CrumbHandle {
  return typeof handle === "object" && handle !== null && "crumb" in handle;
}

export function TopBar() {
  const matches = useMatches();
  const crumbs = matches.filter((m) => hasCrumb(m.handle));
  const { resolvedTheme, setTheme } = useTheme();

  return (
    <header className="flex items-center justify-between border-b px-3">
      <nav className="flex items-center gap-1 text-xs" aria-label="Breadcrumb">
        {crumbs.map((m, i) => {
          const last = i === crumbs.length - 1;
          const label = (m.handle as CrumbHandle).crumb(m.params);
          return (
            <Fragment key={m.id}>
              {i > 0 && <ChevronRight className="h-3 w-3 text-muted-foreground" />}
              {last ? (
                <span className="font-medium">{label}</span>
              ) : (
                <Link to={m.pathname} className="text-muted-foreground hover:text-foreground">
                  {label}
                </Link>
              )}
            </Fragment>
          );
        })}
      </nav>
      <div className="flex items-center gap-2">
        <span
          className="inline-block h-2 w-2 rounded-full bg-muted-foreground/50"
          title={t("ws.disconnected")}
          aria-label={t("ws.disconnected")}
        />
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7"
          aria-label={t("theme.toggle")}
          onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")}
        >
          <Sun className="h-4 w-4 dark:hidden" />
          <Moon className="hidden h-4 w-4 dark:block" />
        </Button>
      </div>
    </header>
  );
}
