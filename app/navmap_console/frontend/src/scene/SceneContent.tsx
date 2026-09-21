import { useMemo } from "react";
import type { Scene } from "@/api/scene-bundle";
import { useSceneStore } from "@/stores/scene-store";
import { computeBounds, medianEdgeLength } from "./build/bounds";
import type { Palette } from "./build/colors";
import { lerpPositions } from "./build/segments";
import { autoNodeStyle, nodeSizeFor } from "./build/visibility";
import { CameraRig } from "./CameraRig";
import { EdgesLayer } from "./layers/EdgesLayer";
import { GhostLayer } from "./layers/GhostLayer";
import { GroundGrid } from "./layers/GroundGrid";
import { LoopsLayer } from "./layers/LoopsLayer";
import { NodesLayer } from "./layers/NodesLayer";
import { SelectionMarker } from "./layers/SelectionMarker";

export function SceneContent({ scene, palette }: { scene: Scene; palette: Palette }) {
  const layers = useSceneStore((s) => s.layers);
  const morph = useSceneStore((s) => s.morph);
  const nodeStyle = useSceneStore((s) => s.nodeStyle);
  const positions = useMemo(() => (morph >= 1 ? scene.pos : lerpPositions(scene.posPre, scene.pos, morph)), [scene, morph]);
  const bounds = useMemo(() => computeBounds(scene.pos), [scene]);
  const median = useMemo(() => medianEdgeLength(scene.pos, scene.odom), [scene]);
  const nodeSize = nodeSizeFor(median, bounds.radius);
  const style = autoNodeStyle(scene.numNodes, nodeStyle);
  return (
    <>
      <CameraRig bounds={bounds} ready />
      <ambientLight intensity={0.85} />
      <directionalLight position={[3, 5, 8]} intensity={0.5} />
      <GroundGrid bounds={bounds} />
      {layers.ghost && <GhostLayer scene={scene} palette={palette} />}
      {layers.odom && <EdgesLayer positions={positions} pairs={scene.odom} color={palette.ref} opacity={0.9} />}
      {layers.covis && <EdgesLayer positions={positions} pairs={scene.covis} color={palette.pins[0]} opacity={0.25} />}
      {layers.trav && <EdgesLayer positions={positions} pairs={scene.trav} color={palette.pins[2]} opacity={0.45} />}
      {layers.loops && <LoopsLayer scene={scene} positions={positions} palette={palette} nodeSize={nodeSize} />}
      <NodesLayer scene={scene} positions={positions} palette={palette} nodeSize={nodeSize} style={style} />
      <SelectionMarker scene={scene} positions={positions} palette={palette} nodeSize={nodeSize} />
    </>
  );
}
