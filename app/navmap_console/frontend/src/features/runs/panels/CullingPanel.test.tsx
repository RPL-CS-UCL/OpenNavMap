import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import { useSceneStore } from "@/stores/scene-store";
import { CullingPanel } from "./CullingPanel";

const wrap = (el: React.ReactNode) =>
  render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>{el}</QueryClientProvider>);

describe("CullingPanel", () => {
  beforeEach(() => useSceneStore.getState().reset());

  it("renders culled rows with thumbnails and method badges", async () => {
    wrap(<CullingPanel rid="reg_1" runId="run_20260918_120000_cd34" step={1} />);
    expect(await screen.findByText("culled_by_forward")).toBeInTheDocument();
    expect(screen.getAllByRole("img").length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText(/0\.91/)).toBeInTheDocument();
  });

  it("clicking a row selects the node and enables the culled layer", async () => {
    wrap(<CullingPanel rid="reg_1" runId="run_20260918_120000_cd34" step={1} />);
    await userEvent.click(await screen.findByTestId("cull-row-23"));
    expect(useSceneStore.getState().selected).toEqual({ kind: "node", id: 23 });
    expect(useSceneStore.getState().layers.culled).toBe(true);
  });

  it("lists kept rows in a collapsed section", async () => {
    wrap(<CullingPanel rid="reg_1" runId="run_20260918_120000_cd34" step={1} />);
    await userEvent.click(await screen.findByRole("button", { name: /kept/i }));
    expect(screen.getByTestId("cull-row-12")).toBeInTheDocument();
  });
});
