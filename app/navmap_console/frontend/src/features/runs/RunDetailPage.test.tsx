import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import { RunDetailPage } from "./RunDetailPage";

vi.mock("@/scene/SceneCanvas", () => ({ SceneCanvas: ({ scene }: { scene: unknown }) => (
  <div data-testid="scene-canvas">{scene ? "scene" : "empty"}</div>
) }));

function wrap(entry = "/regions/reg_1/runs/run_20260918_120000_cd34") {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={[entry]}>
        <Routes>
          <Route path="/regions/:rid/runs/:runId" element={<RunDetailPage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("RunDetailPage", () => {
  it("shows run status, the step table with PGO errors and a cancel button while running", async () => {
    wrap();
    expect(await screen.findByText("first merge")).toBeInTheDocument();
    const row0 = screen.getByRole("row", { name: "step 0" });
    // The session list loads separately, so the session name ("000" for ses_1) may land a tick later.
    await waitFor(() => expect(row0).toHaveTextContent("000"));
    expect(row0).toHaveTextContent("1.234");
    expect(row0).toHaveTextContent("0.456");
    expect(row0).toHaveTextContent("done");
    expect(screen.getByRole("row", { name: "step 1" })).toHaveTextContent("running");
    expect(screen.getByRole("button", { name: "Cancel run" })).toBeEnabled();
    expect(screen.getByText("1 / 2")).toBeInTheDocument();
  });

  it("renders the 3D viewer in the viz tab", async () => {
    wrap();
    await userEvent.click(await screen.findByRole("tab", { name: "3D" }));
    expect(await screen.findByText("1 / 1")).toBeInTheDocument();
    expect(screen.getByTestId("scene-canvas")).toBeInTheDocument();
    expect(screen.queryByText("Interactive 3D, distance matrix")).not.toBeInTheDocument();
  });

  it("defaults to the viz tab when ?step= is present", async () => {
    wrap("/regions/reg_1/runs/run_20260918_120000_cd34?step=0");
    expect(await screen.findByText("0 / 1")).toBeInTheDocument();
  });
});
