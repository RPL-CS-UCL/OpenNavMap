import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, waitFor } from "@testing-library/react";
import { useEffect } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useStepSummaries } from "@/api/hooks/use-results";
import { state } from "@/test/handlers";
import { useLiveSteps } from "./use-live-steps";

const mock = vi.hoisted(() => ({ status: "open" as string, handlers: new Map<string, (m: unknown) => void>() }));

vi.mock("@/ws/use-socket", () => ({
  useSocketStatus: () => mock.status,
  useTopic: (topic: string | null, handler: (m: unknown) => void) => {
    useEffect(() => {
      if (!topic) return;
      const t = topic;
      mock.handlers.set(t, handler);
      return () => { mock.handlers.delete(t); };
    }, [topic]);
  },
}));

const RUN_ID = "run_20260918_120000_cd34";

function Harness({ rid = "reg_1", runId = RUN_ID, pollMs }: { rid?: string; runId?: string; pollMs?: number }) {
  useLiveSteps(rid, runId, { pollMs });
  useStepSummaries(rid, runId);
  return <div data-testid="harness" />;
}

describe("useLiveSteps", () => {
  beforeEach(() => {
    mock.status = "open";
    mock.handlers.clear();
    state.summaryHits = 0;
    state.sceneHits = 0;
  });

  it("invalidates summaries and prefetches the scene on run.step_completed", async () => {
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
    render(<QueryClientProvider client={qc}><Harness /></QueryClientProvider>);
    await waitFor(() => expect(state.summaryHits).toBe(1)); // initial fetch

    const handler = mock.handlers.get(`run:${RUN_ID}`);
    expect(handler).toBeDefined();
    await act(async () => {
      handler!({ topic: `run:${RUN_ID}`, type: "run.step_completed", seq: 1, ts: "", data: { step: { index: 0 } } });
    });
    await waitFor(() => expect(state.summaryHits).toBeGreaterThan(1));
    await waitFor(() => expect(state.sceneHits).toBeGreaterThan(0));
  });

  it("invalidates summaries and the evaluations list on run.evaluated", async () => {
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
    render(<QueryClientProvider client={qc}><Harness /></QueryClientProvider>);
    await waitFor(() => expect(state.summaryHits).toBe(1));
    const spy = vi.spyOn(qc, "invalidateQueries");
    await act(async () => {
      mock.handlers.get(`run:${RUN_ID}`)!({ topic: `run:${RUN_ID}`, type: "run.evaluated", seq: 2, ts: "", data: { kind: "per_step_eval" } });
    });
    await waitFor(() => expect(state.summaryHits).toBe(2));
    expect(spy).toHaveBeenCalledWith({ queryKey: ["regions", "reg_1", "runs", RUN_ID, "evaluations"] });
  });

  it("falls back to polling while the socket is down", async () => {
    mock.status = "closed";
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
    render(<QueryClientProvider client={qc}><Harness pollMs={50} /></QueryClientProvider>);
    await waitFor(() => expect(state.summaryHits).toBeGreaterThan(0)); // initial fetch (polling races it)
    const before = state.summaryHits;
    await waitFor(() => expect(state.summaryHits).toBeGreaterThan(before), { timeout: 3000 });
  });

  it("does not subscribe without a run id", () => {
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
    render(<QueryClientProvider client={qc}><Harness rid="" runId="" /></QueryClientProvider>);
    expect(mock.handlers.size).toBe(0);
  });
});
