import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import { t } from "@/i18n";
import { useSceneStore } from "@/stores/scene-store";
import { summaries } from "@/test/handlers";
import { ateRows, ChartsPanel, edgeSurvivalRows, pgoErrorRows } from "./ChartsPanel";

describe("chart row builders", () => {
  it("pgoErrorRows and edgeSurvivalRows follow the funnel with guards", () => {
    expect(pgoErrorRows(summaries)).toEqual([
      { step: 0, initial: null, final: null },
      { step: 1, initial: 2, final: 0.7 },
    ]);
    const r = edgeSurvivalRows(summaries)[1];
    expect(r.retained).toBe(1);
    expect(r.removedByPgo).toBe(1); // ccm 2 - pgo 1
    expect(r.removedByCcm).toBe(4); // gv 6 - ccm 2
    expect(r.removedByGv).toBe(0); // vpr 6 - gv 6
  });

  it("ateRows passes null through until the step's eval job lands", () => {
    expect(ateRows(summaries)).toEqual([
      { step: 0, ate_trans_rmse: null, ate_rot_rmse: null },
      { step: 1, ate_trans_rmse: 0.612, ate_rot_rmse: 1.23 },
    ]);
  });
});

describe("ChartsPanel", () => {
  beforeEach(() => useSceneStore.getState().reset());

  it("renders six cells; the ATE cell is a chart once any step has a number", () => {
    render(<ChartsPanel summaries={summaries} step={1} />);
    for (const label of ["pgoError", "nodes", "survival", "duration", "newCulled", "ate"]) {
      expect(screen.getByText(label)).toBeInTheDocument();
    }
    expect(screen.queryByText(t("panel.charts.ateNone"))).not.toBeInTheDocument();
  });

  it("shows the no-ATE notice when no step has a number", () => {
    const steps = structuredClone(summaries).map((s) => ({ ...s, ate_trans_rmse: null, ate_rot_rmse: null, ate_frames: null, ate_reason: "no gt" }));
    render(<ChartsPanel summaries={steps} step={1} />);
    expect(screen.getByText(t("panel.charts.ateNone"))).toBeInTheDocument();
  });

  it("clicking a step tick jumps the viewer to that step", async () => {
    useSceneStore.getState().setMaxStep(1);
    useSceneStore.getState().setStep(1);
    render(<ChartsPanel summaries={summaries} step={1} />);
    // every chart renders its own ticks; any tick for step 0 carries the same jump action
    await userEvent.click(screen.getAllByTestId("chart-tick-0")[0]);
    expect(useSceneStore.getState().step).toBe(0);
  });

  it("toggles the table view per cell", async () => {
    render(<ChartsPanel summaries={summaries} step={1} />);
    await userEvent.click(screen.getAllByRole("button", { name: /table/i })[0]);
    expect(screen.getByRole("table")).toBeInTheDocument();
  });
});
