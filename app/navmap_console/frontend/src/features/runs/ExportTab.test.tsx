import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import { Toaster } from "@/components/ui/sonner";
import { t } from "@/i18n";
import { resetState, state } from "@/test/handlers";
import { ExportTab } from "./ExportTab";

const RUN_ID = "run_20260918_120000_cd34";

function wrap() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <ExportTab rid="reg_1" runId={RUN_ID} />
      <Toaster />
    </QueryClientProvider>,
  );
}

describe("ExportTab", () => {
  beforeEach(() => resetState());

  it("lists bundles with size and a download link", async () => {
    wrap();
    expect(await screen.findByText("map_20260918_121500")).toBeInTheDocument();
    expect(screen.getByText("2.5 MiB")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: t("run.export.download") })).toHaveAttribute(
      "href", `/api/regions/reg_1/runs/${RUN_ID}/exports/map_20260918_121500/download`);
  });

  it("creates a bundle and shows it queued without a download link", async () => {
    wrap();
    await userEvent.click(await screen.findByRole("button", { name: t("run.export.create.report") }));
    expect(await screen.findByText("report_20260918_130000")).toBeInTheDocument();
    expect(screen.getByText("queued")).toBeInTheDocument();
    expect(screen.getAllByRole("link", { name: t("run.export.download") })).toHaveLength(1);
    expect(state.exports[0].kind).toBe("report");
  });

  it("deletes a bundle after confirmation", async () => {
    wrap();
    await screen.findByText("map_20260918_121500");
    await userEvent.click(screen.getByRole("button", { name: t("common.delete") }));
    const dialog = await screen.findByRole("dialog");
    await userEvent.click(within(dialog).getByRole("button", { name: t("common.delete") }));
    await waitFor(() => expect(state.exports).toHaveLength(0));
    expect(await screen.findByText(t("run.export.none"))).toBeInTheDocument();
  });

  it("disables delete while a bundle is still packing", async () => {
    state.exports = [{ ...state.exports[0], status: "running", size: undefined }];
    wrap();
    await screen.findByText("map_20260918_121500");
    expect(screen.getByRole("button", { name: t("common.delete") })).toBeDisabled();
    expect(screen.queryByRole("link", { name: t("run.export.download") })).toBeNull();
  });
});
