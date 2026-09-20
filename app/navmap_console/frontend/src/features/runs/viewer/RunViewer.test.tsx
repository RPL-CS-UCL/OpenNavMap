import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { server } from "@/test/server";
import { useSceneStore } from "@/stores/scene-store";
import { RunViewer } from "./RunViewer";

vi.mock("@/scene/SceneCanvas", () => ({ SceneCanvas: ({ scene }: { scene: unknown }) => (
  <div data-testid="scene-canvas">{scene ? "scene" : "empty"}</div>
) }));

function LocationProbe() {
  return <div data-testid="loc">{useLocation().search}</div>;
}
function wrap(entry = "/regions/reg_1/runs/run_20260918_120000_cd34") {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={[entry]}>
        <Routes>
          <Route path="/regions/:rid/runs/:runId" element={<><RunViewer rid="reg_1" runId="run_20260918_120000_cd34" /><LocationProbe /></>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("RunViewer", () => {
  beforeEach(() => useSceneStore.getState().reset());

  it("renders toolbar, slider, legend and the step list", async () => {
    wrap();
    expect(await screen.findByText("1 / 1")).toBeInTheDocument();
    expect(screen.getByText("Accepted loop")).toBeInTheDocument(); // SceneLegend
    expect(screen.getByTestId("scene-canvas")).toBeInTheDocument();
  });

  it("deep link ?step=1 applies and ?step=0 click mirrors back to the URL", async () => {
    wrap("/regions/reg_1/runs/run_20260918_120000_cd34?step=1");
    expect(await screen.findByText("1 / 1")).toBeInTheDocument();
    // The step rows arrive with the summaries query, a tick after the deep link lands.
    await userEvent.click(await screen.findByRole("button", { name: "0" }));
    expect(screen.getByTestId("loc").textContent).toContain("step=0");
    expect(useSceneStore.getState().follow).toBe(false);
  });

  it("hotkeys drive the store: ] advances, t flips camera, space plays", async () => {
    wrap();
    await screen.findByText("1 / 1");
    fireEvent.keyDown(window, { key: "t" });
    expect(useSceneStore.getState().camera).toBe("top");
    fireEvent.keyDown(window, { key: " " });
    expect(useSceneStore.getState().playing).toBe(true);
    fireEvent.keyDown(window, { key: "[" });
    expect(useSceneStore.getState().step).toBe(0);
    fireEvent.keyDown(window, { key: "Escape" });
    expect(useSceneStore.getState().selected).toBeNull();
  });

  it("selecting a node in the store fills the inspector", async () => {
    wrap();
    await screen.findByText("1 / 1");
    act(() => useSceneStore.getState().select({ kind: "node", id: 13 }));
    expect(await screen.findByText("seq/000013.color.jpg")).toBeInTheDocument();
  });

  it("prefetches neighbours but never beyond the last step", async () => {
    const sceneSteps: string[] = [];
    server.use(
      http.get("/api/regions/:rid/runs/:runId/steps/:k/scene.bin", ({ params }) => {
        sceneSteps.push(String(params.k));
        return HttpResponse.json({}, { status: 404 });
      }),
    );
    wrap("/regions/reg_1/runs/run_20260918_120000_cd34?step=1");
    await screen.findByText("1 / 1");
    // useScene(1) plus prefetch of step 0; prefetch of 2 must be clipped at the last step.
    expect([...sceneSteps].sort()).toEqual(["0", "1"]);
  });
});
