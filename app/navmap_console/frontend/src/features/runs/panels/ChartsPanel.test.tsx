import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import { useSceneStore } from "@/stores/scene-store";
import { summaries } from "@/test/handlers";
import { ChartsPanel, edgeSurvivalRows, pgoErrorRows } from "./ChartsPanel";

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
});

describe("ChartsPanel", () => {
  beforeEach(() => useSceneStore.getState().reset());

  it("renders six cells with the ATE placeholder", () => {
    render(<ChartsPanel summaries={summaries} step={1} />);
    for (const label of ["pgoError", "nodes", "survival", "duration", "newCulled", "ate"]) {
      expect(screen.getByText(label)).toBeInTheDocument();
    }
    expect(screen.getByText("ateSoon")).toBeInTheDocument();
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
