import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";
import { prefetchScene } from "@/api/hooks/use-results";
import { qk } from "@/api/query-keys";
import type { WsMessage } from "@/ws/socket-client";
import { useSocketStatus, useTopic } from "@/ws/use-socket";

const POLL_MS = 5000;

export interface LiveStepsOptions {
  /** Polling cadence while the socket is down; injectable so tests can use real timers. */
  pollMs?: number;
}

/**
 * Keep the viewer in step with a live merge run: `run.step_completed` invalidates the summaries
 * (and the events feed, keyed by a shared prefix) and prefetches the new scene; `run.state`
 * invalidates the run detail, `run.evaluated` refreshes the ATE numbers. While the socket is down,
 * poll summaries and run detail every 5 s instead.
 */
export function useLiveSteps(rid: string, runId: string, opts: LiveStepsOptions = {}): void {
  const qc = useQueryClient();
  // Read during render so the latest status reaches the interval callback without a re-render.
  const statusRef = useRef(useSocketStatus());
  statusRef.current = useSocketStatus();

  useTopic(rid && runId ? `run:${runId}` : null, (m: WsMessage) => {
    if (m.type === "run.step_completed") {
      const step = (m.data as { step?: { index?: number } } | undefined)?.step;
      void qc.invalidateQueries({ queryKey: qk.results.summaries(rid, runId) });
      void qc.invalidateQueries({ queryKey: ["regions", rid, "runs", runId, "events"] });
      if (step?.index !== undefined) void prefetchScene(qc, rid, runId, step.index);
    } else if (m.type === "run.state") {
      void qc.invalidateQueries({ queryKey: qk.runs.one(rid, runId) });
    } else if (m.type === "run.evaluated") {
      // an eval job landed: per-step ATE lives in the summaries, the final report in the evaluations list
      void qc.invalidateQueries({ queryKey: qk.results.summaries(rid, runId) });
      void qc.invalidateQueries({ queryKey: qk.evaluations.list(rid, runId) });
    }
  });

  useEffect(() => {
    if (!rid || !runId) return;
    const id = setInterval(() => {
      if (statusRef.current === "open") return;
      void qc.refetchQueries({ queryKey: qk.results.summaries(rid, runId) });
      void qc.refetchQueries({ queryKey: qk.runs.one(rid, runId) });
    }, opts.pollMs ?? POLL_MS);
    return () => clearInterval(id);
  }, [qc, rid, runId, opts.pollMs]);
}
