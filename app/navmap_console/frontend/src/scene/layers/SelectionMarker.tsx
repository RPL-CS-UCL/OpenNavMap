import type { ReactElement } from "react";
import { useMemo } from "react";
import { Color } from "three";
import type { Scene } from "@/api/scene-bundle";
import { type Selection, useSceneStore } from "@/stores/scene-store";
import type { Palette } from "../build/colors";
import { buildSegments } from "../build/segments";
import { nodeRowIndex } from "../build/visibility";
import { RawSegments, noRaycast } from "./EdgesLayer";

interface Props { scene: Scene; positions: Float32Array; palette: Palette; nodeSize: number }

function Ring({ at, radius, color }: { at: [number, number, number]; radius: number; color: Color }) {
  return (
    <mesh position={at} raycast={noRaycast}>
      <sphereGeometry args={[radius, 12, 8]} />
      <meshBasicMaterial color={color} wireframe />
    </mesh>
  );
}

export function SelectionMarker({ scene, positions, palette, nodeSize }: Props) {
  const selected = useSceneStore((s) => s.selected);
  const hovered = useSceneStore((s) => s.hovered);
  const rowOf = useMemo(() => nodeRowIndex(scene.nodeId), [scene]);
  const color = useMemo(() => new Color().setRGB(palette.select[0], palette.select[1], palette.select[2]), [palette]);
  const marks: ReactElement[] = [];
  const add = (sel: Selection, key: string, scale: number) => {
    if (!sel) return;
    if (sel.kind === "node") {
      const row = rowOf.get(sel.id);
      if (row === undefined) return;
      marks.push(<Ring key={key} at={[positions[row * 3], positions[row * 3 + 1], positions[row * 3 + 2]]} radius={nodeSize * scale} color={color} />);
    } else if (sel.index < scene.numLoops) {
      const pairs = scene.loopIdx.subarray(sel.index * 2, sel.index * 2 + 2);
      marks.push(<RawSegments key={key} segments={buildSegments(positions, pairs)} color={palette.select} opacity={1} />);
      for (let k = 0; k < 2; k++) {
        const r = pairs[k];
        marks.push(<Ring key={`${key}-${k}`} at={[positions[r * 3], positions[r * 3 + 1], positions[r * 3 + 2]]} radius={nodeSize * scale * 0.7} color={color} />);
      }
    }
  };
  add(selected, "sel", 0.8);
  if (hovered && JSON.stringify(hovered) !== JSON.stringify(selected)) add(hovered, "hov", 0.55);
  return <>{marks}</>;
}
