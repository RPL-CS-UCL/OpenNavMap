import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import { beforeEach, describe, expect, it } from "vitest";
import { t } from "@/i18n";
import { resetState, state } from "@/test/handlers";
import { server } from "@/test/server";
import { Toaster } from "@/components/ui/sonner";
import { EvaluationTab } from "./EvaluationTab";

function wrap(runId = "run_20260918_120000_cd34") {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <EvaluationTab rid="reg_1" runId={runId} />
      <Toaster />
    </QueryClientProvider>,
  );
}

describe("EvaluationTab", () => {
  beforeEach(() => resetState());

  it("shows the ATE numbers, the job badge and the report files", async () => {
    wrap();
    expect(await screen.findByTestId("ate-trans")).toHaveTextContent("0.612 m");
    expect(screen.getByTestId("ate-rot")).toHaveTextContent("1.23 °");
    expect(screen.getByTestId("ate-frames")).toHaveTextContent("24");
    expect(screen.getByText("succeeded")).toBeInTheDocument();
    const downloads = screen.getAllByRole("link", { name: t("run.evaluation.download") });
    expect(downloads).toHaveLength(2);
    expect(downloads[1]).toHaveAttribute(
      "href",
      "/api/regions/reg_1/runs/run_20260918_120000_cd34/evaluations/final/files/report_benchmark_eval_config/plot.pdf",
    );
    // only image/pdf files get a preview link
    expect(screen.getAllByRole("link", { name: t("run.evaluation.preview") })).toHaveLength(1);
  });

  it("re-run posts to /evaluate and the list shows the queued job", async () => {
    wrap();
    await userEvent.click(await screen.findByRole("button", { name: t("run.evaluation.rerun") }));
    expect(await screen.findByText("queued")).toBeInTheDocument();
    expect(screen.getByText(t("run.evaluation.pending"))).toBeInTheDocument();
    expect(state.evaluations[0].status).toBe("queued");
  });

  it("shows the no-GT empty state and a toast when re-run is refused", async () => {
    state.evaluations = [];
    wrap("run_no_gt");
    expect(await screen.findByText(t("run.evaluation.none"))).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: t("run.evaluation.rerun") }));
    expect(await screen.findByText(/not found/)).toBeInTheDocument();
  });

  it("shows the error of a failed evaluation", async () => {
    state.evaluations = [{ eid: "final", status: "failed", error: "no translation rmse parsed", job: null }];
    server.use(http.get("/api/regions/:rid/runs/:runId/evaluations", () =>
      HttpResponse.json({ items: state.evaluations, evaluating: false })));
    wrap();
    expect(await screen.findByText("no translation rmse parsed")).toBeInTheDocument();
  });
});
