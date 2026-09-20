import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import { decodeSceneBundle, toScene } from "@/api/scene-bundle";
import { t } from "@/i18n";
import { useSceneStore } from "@/stores/scene-store";
import { summaries } from "@/test/handlers";
import { makeSceneFixture } from "@/test/scene-fixture";
import { Inspector } from "./Inspector";
import { PanelDock } from "./PanelDock";
import { StepList } from "./StepList";
import { StepSlider } from "./StepSlider";

const scene = toScene(decodeSceneBundle(makeSceneFixture(24, 3)));
const qcWrap = (el: React.ReactNode) =>
  render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>{el}</QueryClientProvider>);

describe("StepList", () => {
  beforeEach(() => useSceneStore.getState().reset());

  it("renders step rows with session names and node counts", () => {
    useSceneStore.getState().setMaxStep(1);
    render(<StepList steps={summaries} pending={false} />);
    expect(screen.getByRole("button", { name: /^0/ })).toHaveTextContent("ses_1");
    expect(screen.getByText("24")).toBeInTheDocument();
  });

  it("shows the split, degraded and all-rejected markers", () => {
    const steps = structuredClone(summaries);
    steps[1].component_sizes = [20, 4];
    steps[1].pgo_error_final = 5; // 比第 0 步 0.7 差
    steps[1].loops = { total: 3, new: 3, hist: 0, accepted: 0, rejected_new: 3, overturned_hist: 0 };
    useSceneStore.getState().setMaxStep(1);
    render(<StepList steps={steps} pending={false} />);
    expect(screen.getByTitle(t("viewer.steps.marker.split", { n: 2 }))).toBeInTheDocument();
    expect(screen.getByTitle(t("viewer.steps.marker.degraded"))).toBeInTheDocument();
    expect(screen.getByTestId("all-rejected-1")).toBeInTheDocument();
  });

  it("clicking a row selects the step and the follow badge counts lagging steps", async () => {
    useSceneStore.getState().setMaxStep(3);
    render(<StepList steps={summaries} pending={false} />);
    await userEvent.click(screen.getByRole("button", { name: /^0/ }));
    expect(useSceneStore.getState().step).toBe(0);
    expect(screen.getByText(t("viewer.steps.newSteps", { count: 3 }))).toBeInTheDocument();
  });
});

describe("StepSlider", () => {
  beforeEach(() => useSceneStore.getState().reset());

  it("advances the step and disables the morph slider without pre-PGO data", async () => {
    useSceneStore.getState().setMaxStep(1);
    useSceneStore.getState().setStep(0);
    render(<StepSlider hasPrePgo={false} />);
    expect(screen.getByText("0 / 1")).toBeInTheDocument();
    // The slider role lives on a span, which jest-dom's toBeDisabled does not accept.
    expect(screen.getByRole("slider", { name: t("viewer.slider.morph") })).toHaveAttribute("aria-disabled", "true");
    await userEvent.click(screen.getByRole("button", { name: t("viewer.slider.next") }));
    expect(useSceneStore.getState().step).toBe(1);
  });

  it("speed toggle sets the playback speed", async () => {
    useSceneStore.getState().setMaxStep(1);
    render(<StepSlider hasPrePgo />);
    await userEvent.click(screen.getByRole("radio", { name: "×2" }));
    expect(useSceneStore.getState().speed).toBe(2);
  });
});

describe("Inspector", () => {
  beforeEach(() => useSceneStore.getState().reset());

  it("shows the step summary when nothing is selected", async () => {
    useSceneStore.getState().setMaxStep(1);
    qcWrap(<Inspector rid="reg_1" runId="run_20260918_120000_cd34" scene={scene} />);
    // The fixture's dir_name equals its session_id, so the name renders twice.
    expect((await screen.findAllByText("merge_001_b")).length).toBeGreaterThan(0);
    expect(screen.getByText("24")).toBeInTheDocument();
  });

  it("shows node details fetched from the API and pins/flys", async () => {
    useSceneStore.getState().setMaxStep(1);
    useSceneStore.getState().select({ kind: "node", id: 13 });
    qcWrap(<Inspector rid="reg_1" runId="run_20260918_120000_cd34" scene={scene} />);
    expect(await screen.findByText("seq/000013.color.jpg")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: t("viewer.inspector.pinStep", { step: 1 }) }));
    expect(useSceneStore.getState().pinned).toContain(1);
    expect(useSceneStore.getState().flyTo).not.toBeNull();
  });

  it("shows loop factor data from the scene arrays", () => {
    useSceneStore.getState().setMaxStep(1);
    useSceneStore.getState().select({ kind: "loop", index: 0 });
    qcWrap(<Inspector rid="reg_1" runId="run_20260918_120000_cd34" scene={scene} />);
    expect(screen.getByText("12")).toBeInTheDocument(); // query 端 id
    expect(screen.getByText(/0\.9/)).toBeInTheDocument(); // GNC weight
  });
});

describe("PanelDock", () => {
  it("renders six tab triggers with the VPR matrix panel active", async () => {
    qcWrap(<PanelDock rid="reg_1" runId="run_20260918_120000_cd34" step={1} scene={scene} />);
    for (const key of ["viewer.dock.vpr", "viewer.dock.loops", "viewer.dock.pgo", "viewer.dock.culling", "viewer.dock.charts", "viewer.dock.console"] as const) {
      expect(screen.getByRole("tab", { name: t(key) })).toBeInTheDocument();
    }
    expect(await screen.findByRole("img", { name: "D-matrix" })).toBeInTheDocument();
  });
});
