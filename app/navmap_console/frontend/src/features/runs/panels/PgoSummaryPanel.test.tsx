import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import { decodeSceneBundle, toScene } from "@/api/scene-bundle";
import { useSceneStore } from "@/stores/scene-store";
import { summaries } from "@/test/handlers";
import { makeSceneFixture } from "@/test/scene-fixture";
import { PgoSummaryPanel, weightHistogram } from "./PgoSummaryPanel";

const scene = toScene(decodeSceneBundle(makeSceneFixture(24, 3)));

describe("weightHistogram", () => {
  it("bins weights into ten buckets and skips NaN", () => {
    const w = new Float32Array([0.04, 0.15, 0.95, NaN, 1.0]);
    const h = weightHistogram(w, 10);
    expect(h[0]).toBe(1); // 0.04
    expect(h[1]).toBe(1); // 0.15
    expect(h[9]).toBe(2); // 0.95 与 1.0
    expect(h.reduce((a, b) => a + b, 0)).toBe(4);
  });

  it("returns empty bins for null weights", () => {
    expect(weightHistogram(null, 10)).toHaveLength(10);
  });
});

describe("PgoSummaryPanel", () => {
  beforeEach(() => useSceneStore.getState().reset());

  it("shows the error delta and loop stats", () => {
    render(<PgoSummaryPanel summary={summaries[1]} scene={scene} />);
    expect(screen.getByText(/2 → 0\.7/)).toBeInTheDocument();
    expect(screen.getByText("accepted")).toBeInTheDocument();
  });

  it("warns when more than half of historic loops are overturned", () => {
    const s = { ...summaries[1], loops: { total: 4, new: 0, hist: 4, accepted: 2, rejected_new: 0, overturned_hist: 3 } };
    render(<PgoSummaryPanel summary={s} scene={scene} />);
    expect(screen.getByText("overturnedWarn")).toBeInTheDocument();
  });

  it("threshold slider and ghost toggle drive the store", async () => {
    render(<PgoSummaryPanel summary={summaries[1]} scene={scene} />);
    await userEvent.click(screen.getByRole("switch", { name: /ghost/i }));
    expect(useSceneStore.getState().layers.ghost).toBe(true);
    expect(screen.getByRole("slider")).toBeInTheDocument();
  });
});
