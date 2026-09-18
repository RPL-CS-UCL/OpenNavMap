import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import type { ReactElement } from "react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { RegionsPage } from "./RegionsPage";

function wrap(ui: ReactElement) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter>{ui}</MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("RegionsPage", () => {
  it("lists regions from the API with session counts", async () => {
    wrap(<RegionsPage />);
    expect(await screen.findByText("campus")).toBeInTheDocument();
    expect(screen.getByRole("row", { name: /campus/ })).toHaveTextContent("1");
  });
});
