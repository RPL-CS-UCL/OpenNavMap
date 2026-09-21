import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { t } from "@/i18n";
import { resetState, state } from "@/test/handlers";
import { GeoMapPanel } from "./GeoMapPanel";

// jsdom has no layout, so Leaflet is replaced by a recorder of what the panel draws.
const drawn = vi.hoisted(() => ({ polylines: [] as unknown[], markers: 0, removed: 0, fitted: 0 }));
vi.mock("leaflet", () => {
  const layer = () => ({ addTo: () => layer() });
  return {
    default: {
      map: () => ({ fitBounds: () => { drawn.fitted += 1; }, remove: () => { drawn.removed += 1; } }),
      tileLayer: layer,
      polyline: (pts: unknown) => { drawn.polylines.push(pts); return { addTo: () => ({ getBounds: () => null }) }; },
      circleMarker: () => { drawn.markers += 1; return layer(); },
    },
  };
});

const RUN_ID = "run_20260918_120000_cd34";

function wrap() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <GeoMapPanel rid="reg_1" runId={RUN_ID} />
    </QueryClientProvider>,
  );
}

describe("GeoMapPanel", () => {
  beforeEach(() => {
    resetState();
    drawn.polylines = [];
    drawn.markers = 0;
    drawn.removed = 0;
    drawn.fitted = 0;
  });

  it("defaults to the last step and draws the trajectory and the GPS fixes", async () => {
    wrap();
    expect(await screen.findByTestId("geo-map")).toBeInTheDocument();
    expect(screen.getByTestId("geo-stats")).toHaveTextContent("2 of 3 frames have a GPS fix");
    expect(screen.getByTestId("geo-stats")).toHaveTextContent("fit residual 4.2 m");
    expect(screen.getByRole("combobox", { name: t("run.map.step") })).toHaveValue("1");
    expect(drawn.polylines).toHaveLength(1);
    expect(drawn.markers).toBe(2);
    expect(drawn.fitted).toBe(1);
  });

  it("switching the step fetches again and shows the no-GPS state", async () => {
    wrap();
    await screen.findByTestId("geo-map");
    await userEvent.selectOptions(screen.getByRole("combobox", { name: t("run.map.step") }), "0");
    expect(await screen.findByText(t("run.map.noGps"))).toBeInTheDocument();
    await waitFor(() => expect(state.geoHits).toBe(2));
    expect(drawn.removed).toBe(1); // the previous map was torn down
  });

  it("shows an empty state when the run has no completed step", async () => {
    state.summaries = [];
    wrap();
    expect(await screen.findByText(t("run.map.noSteps"))).toBeInTheDocument();
  });
});
