import { ArrowLeft } from "lucide-react";
import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { t } from "@/i18n";

interface Props {
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
  backTo?: string;
}

export function PageHeader({ title, description, actions, backTo }: Props) {
  return (
    <div className="mb-4 flex items-start justify-between gap-4">
      <div className="min-w-0">
        {backTo && (
          <Button asChild variant="link" size="sm" className="h-auto px-0 text-xs text-muted-foreground">
            <Link to={backTo}>
              <ArrowLeft className="mr-1 h-3 w-3" />
              {t("common.back")}
            </Link>
          </Button>
        )}
        <h1 className="truncate text-lg font-semibold leading-7">{title}</h1>
        {description && <div className="text-xs text-muted-foreground">{description}</div>}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </div>
  );
}
