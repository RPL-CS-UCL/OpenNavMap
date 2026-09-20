import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it } from "vitest";
import { decodeSceneBundle, toScene } from "@/api/scene-bundle";
import { useSceneStore } from "@/stores/scene-store";
import { server } from "@/test/server";
import { makeSceneFixture } from "@/test/scene-fixture";
import { DMatrixPanel } from "./DMatrixPanel";
import { findLoopIndex, nearestCandidate, zoomAtPoint } from "./dmatrix-math";

const scene = toScene(decodeSceneBundle(makeSceneFixture(24, 3)));
const candidates = [
  { db: 0, query: 12, stage: "vpr", gv_inliers: 0 },
  { db: 1, query: 13, stage: "gv", gv_inliers: 40 },
];
const wrap = (el: React.ReactNode) =>
  render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>{el}</QueryClientProvider>);

describe("matrix math", () => {
  it("zoomAtPoint anchors the cursor under the wheel", () => {
    const r = zoomAtPoint(1, { x: 0, y: 0 }, { x: 100, y: 50 }, 2);
    expect(r.zoom).toBe(2);
    expect(r.pan.x).toBe(-100); // 100 + (0-100)*2
  });

  it("nearestCandidate finds the factor within radius cells", () => {
    const rowOf = new Map([[0, 0], [1, 1]]);
    const colOf = new Map([[12, 0], [13, 1]]);
    expect(nearestCandidate(0, 0, candidates, rowOf, colOf, 1.5)?.db).toBe(0);
    expect(nearestCandidate(5, 5, candidates, rowOf, colOf, 1.5)).toBeNull();
  });

  it("findLoopIndex matches both orientations", () => {
    expect(findLoopIndex(scene, 0, 12)).toBe(0);
    expect(findLoopIndex(scene, 12, 0)).toBe(0);
    expect(findLoopIndex(scene, 0, 99)).toBe(-1);
  });
});

describe("DMatrixPanel", () => {
  beforeEach(() => useSceneStore.getState().reset());

  it("renders the image with candidate and factor markers", async () => {
    wrap(<DMatrixPanel rid="reg_1" runId="run_20260918_120000_cd34" step={1} scene={scene} />);
    expect(await screen.findByRole("img", { name: "D-matrix" })).toBeInTheDocument();
    expect(screen.getAllByTestId("candidate-marker")).toHaveLength(2);
    expect(screen.getByTestId("factor-marker-0-12")).toBeInTheDocument();
  });

  it("shows the legacy badge without overlays when dmatrix.json 404s", async () => {
    server.use(http.get("/api/regions/:rid/runs/:runId/steps/:k/dmatrix.json", () => HttpResponse.json({ detail: "no npy" }, { status: 404 })));
    wrap(<DMatrixPanel rid="reg_1" runId="run_20260918_120000_cd34" step={1} scene={scene} />);
    expect(await screen.findByText("legacy image")).toBeInTheDocument();
    expect(screen.queryByTestId("candidate-marker")).not.toBeInTheDocument();
  });

  it("clicking a cell opens the pair card and selects the loop in the scene", async () => {
    wrap(<DMatrixPanel rid="reg_1" runId="run_20260918_120000_cd34" step={1} scene={scene} />);
    fireEvent.click(await screen.findByTestId("matrix-overlay"), { clientX: 10, clientY: 10 });
    expect(await screen.findByText("Loop pair 0 · 12")).toBeInTheDocument();
    expect(useSceneStore.getState().selected).toEqual({ kind: "loop", index: 0 });
  });

  it("arrow keys cycle through candidates", async () => {
    wrap(<DMatrixPanel rid="reg_1" runId="run_20260918_120000_cd34" step={1} scene={scene} />);
    await screen.findByRole("img", { name: "D-matrix" });
    fireEvent.keyDown(window, { key: "ArrowRight" });
    expect(useSceneStore.getState().selected).toEqual({ kind: "loop", index: 1 });
  });
});
