import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it } from "vitest";
import { state } from "@/test/handlers";
import { RunWizardPage } from "./RunWizardPage";
import { useWizardStore } from "./wizard-store";

function wrap() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={["/regions/reg_1/runs/new"]}>
        <Routes>
          <Route path="/regions/:rid/runs/new" element={<RunWizardPage />} />
          <Route path="/regions/:rid/runs/:runId" element={<div>run page</div>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("RunWizardPage", () => {
  beforeEach(() => {
    useWizardStore.getState().reset();
    state.runs[0].status = "succeeded"; // otherwise the region counts as busy
  });

  it("walks sessions -> mode -> params -> review and creates a run", async () => {
    wrap();
    const cb = await screen.findByRole("checkbox", { name: state.sessions[0].name });
    await userEvent.click(cb);
    await userEvent.click(screen.getByRole("button", { name: "Next" }));
    expect(screen.getByRole("radio", { name: "Fresh merge" })).toBeChecked();
    expect(screen.getByRole("radio", { name: "Append to current final map" })).toBeDisabled();
    await userEvent.click(screen.getByRole("button", { name: "Next" }));
    await screen.findByLabelText("pgo_loop_sigma_trans");
    await userEvent.click(screen.getByRole("button", { name: "Next" }));
    expect(screen.getByText("1 session")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Start run" }));
    await waitFor(() => expect(screen.getByText("run page")).toBeInTheDocument());
  });
});
