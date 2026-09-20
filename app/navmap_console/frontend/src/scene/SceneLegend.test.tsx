import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { decodeSceneBundle, toScene } from "@/api/scene-bundle";
import { useSceneStore } from "@/stores/scene-store";
import { makeSceneFixture } from "@/test/scene-fixture";
import { SceneLegend } from "./SceneLegend";

const scene = toScene(decodeSceneBundle(makeSceneFixture(24, 3)));

describe("SceneLegend", () => {
  beforeEach(() => useSceneStore.getState().reset());

  it("focus mode lists reference, current and pinned steps", () => {
    useSceneStore.getState().setMaxStep(1);
    useSceneStore.getState().togglePin(0);
    render(<SceneLegend scene={scene} />);
    expect(screen.getByText("Reference")).toBeInTheDocument();
    expect(screen.getByText("Step 1 (current)")).toBeInTheDocument();
    expect(screen.getByText("Step 0 (pinned)")).toBeInTheDocument();
  });

  it("component mode counts nodes per component", () => {
    useSceneStore.getState().setColorMode("component");
    render(<SceneLegend scene={scene} />);
    expect(screen.getByText("Component 0 · 24 nodes")).toBeInTheDocument();
  });

  it("order mode shows a gradient bar with the step range", () => {
    useSceneStore.getState().setMaxStep(5);
    useSceneStore.getState().setColorMode("order");
    render(<SceneLegend scene={scene} />);
    expect(screen.getByTestId("legend-gradient")).toBeInTheDocument();
    expect(screen.getByText("step 0")).toBeInTheDocument();
    expect(screen.getByText("step 5")).toBeInTheDocument();
  });
});
