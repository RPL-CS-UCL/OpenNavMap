import { AlertTriangle } from "lucide-react";
import { Link, isRouteErrorResponse, useRouteError } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { t } from "@/i18n";
import { EmptyState } from "./EmptyState";

export function RouteError() {
  const err = useRouteError();
  const detail = isRouteErrorResponse(err)
    ? `${err.status} ${err.statusText}`
    : err instanceof Error
      ? err.message
      : String(err);
  return (
    <div className="p-4">
      <EmptyState
        icon={AlertTriangle}
        title={t("common.error", { detail })}
        action={
          <Button asChild variant="outline" size="sm">
            <Link to="/regions">{t("common.back")}</Link>
          </Button>
        }
      />
    </div>
  );
}
