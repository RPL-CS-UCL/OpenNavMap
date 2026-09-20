import { Canvas } from "@react-three/fiber";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Color, type WebGLRenderer } from "three";
import type { Scene } from "@/api/scene-bundle";
import { Button } from "@/components/ui/button";
import { t } from "@/i18n";
import { cn } from "@/lib/utils";
import { useSceneStore } from "@/stores/scene-store";
import { SceneContent } from "./SceneContent";
import { SceneErrorBoundary } from "./SceneErrorBoundary";
import { useScenePalette } from "./use-scene-palette";

interface Props { scene: Scene | null; className?: string }

export function SceneCanvas({ scene, className }: Props) {
  const { palette, background } = useScenePalette();
  const [lost, setLost] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [gl, setGl] = useState<WebGLRenderer | null>(null);
  const select = useSceneStore((s) => s.select);
  const clear = useMemo(() => new Color(background), [background]);

  useEffect(() => { gl?.setClearColor(clear); }, [gl, clear]);
  useEffect(() => {
    if (!gl) return;
    const onLost = (e: Event) => { e.preventDefault(); setLost(true); };
    gl.domElement.addEventListener("webglcontextlost", onLost);
    return () => gl.domElement.removeEventListener("webglcontextlost", onLost);
  }, [gl]);
  const reload = useCallback(() => { setLost(false); setEpoch((n) => n + 1); }, []);

  return (
    <div className={cn("relative h-full w-full", className)} style={{ background }} data-testid="scene-canvas">
      <SceneErrorBoundary onReset={reload}>
        <Canvas key={epoch} dpr={[1, 2]} gl={{ antialias: true, powerPreference: "high-performance" }}
          onCreated={({ gl: renderer }) => setGl(renderer)} onPointerMissed={() => select(null)}>
          {scene && <SceneContent scene={scene} palette={palette} />}
        </Canvas>
      </SceneErrorBoundary>
      {!scene && <div className="absolute inset-0 flex items-center justify-center text-sm text-muted-foreground">{t("scene.empty")}</div>}
      {lost && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-background/80 text-sm">
          <span>{t("scene.contextLost")}</span>
          <Button size="sm" onClick={reload}>{t("scene.reload")}</Button>
        </div>
      )}
    </div>
  );
}
