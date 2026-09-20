import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it } from "vitest";
import { server } from "@/test/server";
import { state } from "@/test/handlers";
import { Toaster } from "@/components/ui/sonner";
import { ImportRunDialog } from "./ImportRunDialog";

function wrap() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={["/regions/reg_1"]}>
        <Routes>
          <Route path="/regions/:rid" element={<ImportRunDialog open onOpenChange={() => {}} />} />
          <Route path="/regions/:rid/runs/:runId" element={<div data-testid="run-page" />} />
        </Routes>
      </MemoryRouter>
      <Toaster />
    </QueryClientProvider>,
  );
}

describe("ImportRunDialog", () => {
  beforeEach(() => { state.counter = 0; });

  it("submits the body and navigates to the imported run", async () => {
    wrap();
    await userEvent.type(screen.getByLabelText(/result directory/i), "/Titan/dataset/results_x");
    await userEvent.type(screen.getByLabelText(/sessions root/i), "/Titan/dataset/sessions_390");
    await userEvent.click(screen.getByRole("switch"));
    await userEvent.click(screen.getByRole("button", { name: /import run/i }));
    expect(await screen.findByTestId("run-page")).toBeInTheDocument();
    const created = state.runs.find((r) => r.kind === "imported");
    expect(created).toBeTruthy();
    expect(created?.name).toBe("results_x");
    expect(state.sessions.filter((s) => s.source === "imported")).toHaveLength(3);
    expect(state.regions[0].head?.run_id).toBe(created?.id);
  });

  it("disables submit until a result directory is set", async () => {
    wrap();
    expect(screen.getByRole("button", { name: /import run/i })).toBeDisabled();
    await userEvent.type(screen.getByLabelText(/result directory/i), "/Titan/dataset/results_x");
    expect(screen.getByRole("button", { name: /import run/i })).toBeEnabled();
  });

  it("surfaces a 403 error as a toast", async () => {
    server.use(http.post("/api/regions/:rid/runs/import", () => HttpResponse.json({ detail: "outside roots" }, { status: 403 })));
    wrap();
    await userEvent.type(screen.getByLabelText(/result directory/i), "/etc");
    await userEvent.click(screen.getByRole("button", { name: /import run/i }));
    expect(await screen.findByText(/outside roots/i)).toBeInTheDocument();
  });
});
