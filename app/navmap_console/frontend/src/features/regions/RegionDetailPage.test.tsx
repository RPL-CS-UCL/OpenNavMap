import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import { RegionDetailPage } from "./RegionDetailPage";

vi.mock("@/scene/SceneCanvas", () => ({ SceneCanvas: () => <div data-testid="scene-canvas" /> }));

function wrap() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={["/regions/reg_1"]}>
        <Routes>
          <Route path="/regions/:rid" element={<RegionDetailPage />} />
          <Route path="/regions/:rid/runs/:runId" element={<div data-testid="run-page" />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("RegionDetailPage import", () => {
  it("opens the import dialog from the Import run button", async () => {
    wrap();
    await screen.findByText("campus");
    await userEvent.click(screen.getByRole("button", { name: /import run/i }));
    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByLabelText(/result directory/i)).toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: /import run/i })).toBeInTheDocument();
  });

  it("submits the dialog and navigates to the imported run", async () => {
    wrap();
    await screen.findByText("campus");
    await userEvent.click(screen.getByRole("button", { name: /import run/i }));
    const dialog = await screen.findByRole("dialog");
    await userEvent.type(within(dialog).getByLabelText(/result directory/i), "/Titan/dataset/results_x");
    await userEvent.click(within(dialog).getByRole("button", { name: /import run/i }));
    expect(await screen.findByTestId("run-page")).toBeInTheDocument();
  });
});
