import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

export type Tone = "ok" | "warn" | "error" | "muted";

const tones: Record<Tone, string> = {
  ok: "border-emerald-600/40 bg-emerald-500/10 text-emerald-800 dark:text-emerald-300",
  warn: "border-amber-600/40 bg-amber-500/10 text-amber-800 dark:text-amber-300",
  error: "border-red-600/40 bg-red-500/10 text-red-800 dark:text-red-300",
  muted: "border-border bg-muted text-muted-foreground",
};

interface Props {
  tone: Tone;
  icon?: LucideIcon;
  children: ReactNode;
  className?: string;
  title?: string;
}

export function StatusBadge({ tone, icon: Icon, children, className, title }: Props) {
  return (
    <span
      title={title}
      className={cn(
        "inline-flex items-center gap-1 rounded-sm border px-1.5 py-0.5 text-[11px] font-medium leading-4",
        tones[tone],
        className,
      )}
    >
      {Icon && <Icon className="h-3 w-3" aria-hidden />}
      {children}
    </span>
  );
}
