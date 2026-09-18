import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import type { ReactElement } from "react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { JobsPage } from "./JobsPage";

function wrap(ui: ReactElement) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter>{ui}</MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("JobsPage", () => {
  it("lists jobs with kind, status and progress", async () => {
    wrap(<JobsPage />);
    const row = await screen.findByRole("row", { name: /job_20260918_120000_ab12/ });
    expect(row).toHaveTextContent("merge");
    expect(row).toHaveTextContent("running");
    expect(row).toHaveTextContent("1 / 2");
    expect(row).toHaveTextContent("pose_graph_optimization");
  });
});
