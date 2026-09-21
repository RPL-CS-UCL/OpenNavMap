import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { useEffect, useRef, useState } from "react";
import { errorDetail } from "@/api/client";
import { useGeo, useStepSummaries } from "@/api/hooks/use-results";
import type { Geo } from "@/api/types";
import { EmptyState } from "@/components/common/EmptyState";
import { Skeleton } from "@/components/ui/skeleton";
import { t } from "@/i18n";

const OSM_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png";
const OSM_ATTRIBUTION = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';

/** Trajectory (line) and raw GPS fixes (dots) drawn on OpenStreetMap; the map is rebuilt when the step changes. */
function LeafletMap({ geo }: { geo: Geo }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    const map = L.map(ref.current, { zoomControl: true });
    L.tileLayer(OSM_URL, { attribution: OSM_ATTRIBUTION, maxZoom: 19 }).addTo(map);
    const line = L.polyline(geo.traj, { color: "#2563eb", weight: 3 }).addTo(map);
    for (const p of geo.gps) L.circleMarker(p, { radius: 2, color: "#dc2626", fillOpacity: 0.8, stroke: false }).addTo(map);
    map.fitBounds(line.getBounds(), { padding: [16, 16] });
    return () => {
      map.remove();
    };
  }, [geo]);
  return <div ref={ref} className="min-h-0 flex-1" data-testid="geo-map" />;
}

export function GeoMapPanel({ rid, runId }: { rid: string; runId: string }) {
  const summaries = useStepSummaries(rid, runId);
  const steps = summaries.data ?? [];
  const [picked, setPicked] = useState<number | null>(null);
  const k = picked ?? (steps.length > 0 ? steps[steps.length - 1].index : null);
  const geo = useGeo(rid, runId, k);

  if (summaries.isPending || (k !== null && geo.isPending)) return <Skeleton className="m-3 h-32" />;
  if (k === null) return <EmptyState title={t("run.map.noSteps")} />;
  if (geo.isError) return <EmptyState title={t("common.error", { detail: errorDetail(geo.error) })} />;
  const data = geo.data!;
  return (
    <div className="flex h-full min-h-[60vh] flex-col gap-2">
      <div className="flex flex-wrap items-center gap-3 text-xs">
        <label className="flex items-center gap-1">
          {t("run.map.step")}
          <select
            aria-label={t("run.map.step")}
            className="h-6 rounded-sm border bg-background px-1"
            value={k}
            onChange={(e) => setPicked(Number(e.target.value))}
          >
            {steps.map((s) => <option key={s.index} value={s.index}>{s.index}</option>)}
          </select>
        </label>
        <span className="font-mono tabular-nums" data-testid="geo-stats">
          {t("run.map.stats", { gps: data.n_gps, frames: data.n_frames })}
          {data.rmse_m !== null && ` · ${t("run.map.rmse", { m: data.rmse_m.toFixed(1) })}`}
        </span>
        <span className="text-muted-foreground">{t("run.map.tiles")}</span>
      </div>
      {data.reason === "no_gps" || data.traj.length === 0 ? (
        <EmptyState title={t("run.map.noGps")} body={t("run.map.noGpsBody")} />
      ) : (
        <LeafletMap geo={data} />
      )}
    </div>
  );
}
