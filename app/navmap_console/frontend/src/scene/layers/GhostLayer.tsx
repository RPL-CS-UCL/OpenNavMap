import { useEffect, useMemo } from "react";
import { BufferAttribute, BufferGeometry, PointsMaterial } from "three";
import type { Scene } from "@/api/scene-bundle";
import { useSceneStore } from "@/stores/scene-store";
import type { Palette } from "../build/colors";
import { buildDisplacements } from "../build/segments";
import { RawSegments, noRaycast } from "./EdgesLayer";

/** Pre-PGO positions as faint points plus pre->post displacement strokes for nodes that moved. */
export function GhostLayer({ scene, palette }: { scene: Scene; palette: Palette }) {
  const threshold = useSceneStore((s) => s.morphThreshold);
  const disp = useMemo(() => buildDisplacements(scene.pos, scene.posPre, threshold), [scene, threshold]);
  const geometry = useMemo(() => {
    const g = new BufferGeometry();
    g.setAttribute("position", new BufferAttribute(scene.posPre, 3));
    return g;
  }, [scene]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  const material = useMemo(() => new PointsMaterial({ size: 4, sizeAttenuation: false, transparent: true, opacity: 0.5 }), []);
  useEffect(() => () => material.dispose(), [material]);
  useEffect(() => { material.color.setRGB(palette.ghost[0], palette.ghost[1], palette.ghost[2]); }, [material, palette]);
  return (
    <>
      <points geometry={geometry} material={material} frustumCulled={false} raycast={noRaycast} />
      <RawSegments segments={disp.segments} color={palette.ghost} opacity={0.9} />
    </>
  );
}
