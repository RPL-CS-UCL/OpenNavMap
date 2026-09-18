import { Ban, Check, CircleDashed, Loader2, Unplug, X } from "lucide-react";
import type { JobProgress, JobStatus } from "@/api/types";
import { StatusBadge, type Tone } from "@/components/common/StatusBadge";
import { type MessageKey, t } from "@/i18n";

const map: Record<JobStatus, { tone: Tone; icon: typeof Check }> = {
  queued: { tone: "muted", icon: CircleDashed },
  running: { tone: "warn", icon: Loader2 },
  succeeded: { tone: "ok", icon: Check },
  failed: { tone: "error", icon: X },
  cancelled: { tone: "muted", icon: Ban },
  orphaned: { tone: "error", icon: Unplug },
};

export function JobStatusBadge({ status }: { status: JobStatus }) {
  const { tone, icon } = map[status];
  return (
    <StatusBadge tone={tone} icon={icon}>
      {status}
    </StatusBadge>
  );
}

const crashKeys: Record<string, MessageKey> = {
  segfault: "jobs.crash.segfault",
  bit_flip: "jobs.crash.bit_flip",
  cuda_oom: "jobs.crash.cuda_oom",
  oom: "jobs.crash.oom",
  terminated: "jobs.crash.terminated",
  error: "jobs.crash.error",
};

export function crashHint(kind: string | null): string | null {
  if (!kind) return null;
  const key = crashKeys[kind];
  return key ? t(key) : null;
}

export function progressText(p: JobProgress): string {
  if (p.total === null) return p.step === null ? "" : String(p.step);
  return `${p.completed_steps} / ${p.total}`;
}
