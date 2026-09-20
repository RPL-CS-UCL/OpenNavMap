import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import { decodeSceneBundle, toScene } from "@/api/scene-bundle";
import { useSceneStore } from "@/stores/scene-store";
import { makeSceneFixture } from "@/test/scene-fixture";
import { LoopEdgesTable } from "./LoopEdgesTable";

const scene = toScene(decodeSceneBundle(makeSceneFixture(24, 3)));

describe("LoopEdgesTable", () => {
  beforeEach(() => useSceneStore.getState().reset());

  it("lists loop rows with origin and status badges", () => {
    render(<LoopEdgesTable rid="reg_1" runId="run_20260918_120000_cd34" scene={scene} initialItemCount={3} />);
    expect(screen.getAllByRole("row").length).toBeGreaterThan(3);
    expect(screen.getByText("accepted")).toBeInTheDocument();
    expect(screen.getAllByText("rejected").length).toBeGreaterThan(0);
  });

  it("clicking a column header sorts by conf", async () => {
    render(<LoopEdgesTable rid="reg_1" runId="run_20260918_120000_cd34" scene={scene} initialItemCount={3} />);
    await userEvent.click(screen.getByRole("button", { name: /conf/i }));
    const cells = screen.getAllByTestId("cell-conf").map((c) => Number(c.textContent));
    expect([...cells].sort((a, b) => b - a)).toEqual(cells);
  });

  it("hover and click sync the 3D selection and fly the camera", () => {
    render(<LoopEdgesTable rid="reg_1" runId="run_20260918_120000_cd34" scene={scene} initialItemCount={3} />);
    fireEvent.mouseEnter(screen.getByTestId("loop-row-0"));
    expect(useSceneStore.getState().hovered).toEqual({ kind: "loop", index: 0 });
    fireEvent.click(screen.getByTestId("loop-row-0"));
    expect(useSceneStore.getState().selected).toEqual({ kind: "loop", index: 0 });
    expect(useSceneStore.getState().flyTo).not.toBeNull();
  });

  it("double-click opens the pair card", async () => {
    render(<LoopEdgesTable rid="reg_1" runId="run_20260918_120000_cd34" scene={scene} initialItemCount={3} />);
    fireEvent.doubleClick(screen.getByTestId("loop-row-0"));
    expect(screen.getByText("Loop pair 0 · 12")).toBeInTheDocument();
  });
});
