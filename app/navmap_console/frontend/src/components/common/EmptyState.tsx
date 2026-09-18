import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

interface Props {
  icon?: LucideIcon;
  title: string;
  body?: string;
  action?: ReactNode;
}

export function EmptyState({ icon: Icon, title, body, action }: Props) {
  return (
    <div className="flex flex-col items-start gap-2 rounded-sm border border-dashed p-6 text-sm">
      {Icon && <Icon className="h-5 w-5 text-muted-foreground" />}
      <div className="font-medium">{title}</div>
      {body && <p className="max-w-prose text-muted-foreground">{body}</p>}
      {action && <div className="pt-1">{action}</div>}
    </div>
  );
}
