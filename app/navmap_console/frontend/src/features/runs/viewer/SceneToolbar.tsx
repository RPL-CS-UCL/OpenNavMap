import { Maximize2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Toggle } from "@/components/ui/toggle";
import { t } from "@/i18n";
import { type ColorMode, type LayerKey, type LoopFilter, type NodeStyle, useSceneStore } from "@/stores/scene-store";

const COLORS: ColorMode[] = ["focus", "order", "component", "new"];
const NODE_STYLES: NodeStyle[] = ["auto", "frustum", "points"];
const LAYERS: LayerKey[] = ["odom", "covis", "trav", "loops", "rejected", "ghost", "culled"];
const LOOP_FILTERS: LoopFilter[] = ["all", "accepted", "rejected", "new", "hist"];

export function SceneToolbar() {
  const camera = useSceneStore((s) => s.camera);
  const colorMode = useSceneStore((s) => s.colorMode);
  const nodeStyle = useSceneStore((s) => s.nodeStyle);
  const loopFilter = useSceneStore((s) => s.loopFilter);
  const layers = useSceneStore((s) => s.layers);
  const up = useSceneStore((s) => s.up);
  const toggleCamera = useSceneStore((s) => s.toggleCamera);
  const setColorMode = useSceneStore((s) => s.setColorMode);
  const setNodeStyle = useSceneStore((s) => s.setNodeStyle);
  const setLoopFilter = useSceneStore((s) => s.setLoopFilter);
  const toggleLayer = useSceneStore((s) => s.toggleLayer);
  const setUp = useSceneStore((s) => s.setUp);
  const requestFit = useSceneStore((s) => s.requestFit);

  return (
    <div className="flex h-8 shrink-0 items-center gap-2 border-b px-2 text-[11px]">
      <Toggle pressed={camera === "top"} onPressedChange={toggleCamera} size="sm" aria-label={t("viewer.toolbar.camera")}>
        {t("viewer.toolbar.camera")}
      </Toggle>
      <Select value={colorMode} onValueChange={(v) => setColorMode(v as ColorMode)}>
        <SelectTrigger className="h-6 w-28" aria-label={t("viewer.toolbar.color")}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {COLORS.map((m) => (
            <SelectItem key={m} value={m}>{t(`viewer.toolbar.color.${m}`)}</SelectItem>
          ))}
        </SelectContent>
      </Select>
      <Select value={nodeStyle} onValueChange={(v) => setNodeStyle(v as NodeStyle)}>
        <SelectTrigger className="h-6 w-28" aria-label={t("viewer.toolbar.nodeStyle")}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {NODE_STYLES.map((m) => (
            <SelectItem key={m} value={m}>{t(`viewer.toolbar.nodeStyle.${m}`)}</SelectItem>
          ))}
        </SelectContent>
      </Select>
      {LAYERS.map((key) => (
        <Toggle key={key} pressed={layers[key]} onPressedChange={() => toggleLayer(key)} size="sm" aria-label={t(`viewer.toolbar.layer.${key}`)}>
          {t(`viewer.toolbar.layer.${key}`)}
        </Toggle>
      ))}
      <Select value={loopFilter} onValueChange={(v) => setLoopFilter(v as LoopFilter)}>
        <SelectTrigger className="h-6 w-24" aria-label={t("viewer.toolbar.loopFilter")}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {LOOP_FILTERS.map((m) => (
            <SelectItem key={m} value={m}>{t(`viewer.toolbar.loopFilter.${m}`)}</SelectItem>
          ))}
        </SelectContent>
      </Select>
      <Toggle pressed={up === "y"} onPressedChange={(on) => setUp(on ? "y" : "z")} size="sm" aria-label={t("viewer.toolbar.up")}>
        {t("viewer.toolbar.up")}
      </Toggle>
      <Button variant="ghost" size="icon" className="h-7 w-7" aria-label={t("viewer.toolbar.fit")} onClick={requestFit}>
        <Maximize2 />
      </Button>
    </div>
  );
}
