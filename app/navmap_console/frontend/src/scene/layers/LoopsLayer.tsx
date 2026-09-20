import { type ThreeEvent, useThree } from "@react-three/fiber";
import { useEffect, useMemo } from "react";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
import { LineSegments2 } from "three/examples/jsm/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";
import type { Scene } from "@/api/scene-bundle";
import { useSceneStore } from "@/stores/scene-store";
import { type Palette, loopColor } from "../build/colors";
import { buildSegments, crossSegments, midpoints } from "../build/segments";
import { loopGroups } from "../build/visibility";
import { RawSegments } from "./EdgesLayer";

interface Props { scene: Scene; positions: Float32Array; palette: Palette; nodeSize: number }
interface GroupProps extends Props { indices: Uint32Array; width: number; dashed: boolean; onHover: (i: number | null) => void; onPick: (i: number) => void }

function pairsOf(scene: Scene, indices: Uint32Array): Uint32Array {
  const pairs = new Uint32Array(indices.length * 2);
  indices.forEach((li, i) => { pairs[i * 2] = scene.loopIdx[li * 2]; pairs[i * 2 + 1] = scene.loopIdx[li * 2 + 1]; });
  return pairs;
}

function LoopGroup({ scene, positions, palette, nodeSize, indices, width, dashed, onHover, onPick }: GroupProps) {
  const size = useThree((s) => s.size);
  const material = useMemo(() => new LineMaterial({ vertexColors: true, linewidth: width, dashed, transparent: true, opacity: 0.95 }), [width, dashed]);
  useEffect(() => () => material.dispose(), [material]);
  useEffect(() => { material.resolution.set(size.width, size.height); }, [material, size.width, size.height]);
  useEffect(() => { material.dashSize = nodeSize * 0.8; material.gapSize = nodeSize * 0.5; material.needsUpdate = true; }, [material, nodeSize]);
  const line = useMemo(() => {
    const colors = new Float32Array(indices.length * 6);
    indices.forEach((li, i) => {
      const c = loopColor(scene.loopFlags[li], scene.loopWeight[li], palette);
      colors.set(c, i * 6);
      colors.set(c, i * 6 + 3);
    });
    const geom = new LineSegmentsGeometry();
    geom.setPositions(buildSegments(positions, pairsOf(scene, indices)));
    geom.setColors(colors);
    const obj = new LineSegments2(geom, material);
    obj.computeLineDistances();
    obj.frustumCulled = false;
    return obj;
  }, [indices, scene, positions, palette, material]);
  useEffect(() => () => line.geometry.dispose(), [line]);
  if (indices.length === 0) return null;
  const at = (e: ThreeEvent<PointerEvent> | ThreeEvent<MouseEvent>) => indices[e.faceIndex ?? 0];
  return (
    <primitive object={line}
      onPointerMove={(e: ThreeEvent<PointerEvent>) => { e.stopPropagation(); onHover(at(e)); }}
      onPointerOut={() => onHover(null)}
      onClick={(e: ThreeEvent<MouseEvent>) => { e.stopPropagation(); onPick(at(e)); }} />
  );
}

export function LoopsLayer(props: Props) {
  const { scene, positions, palette, nodeSize } = props;
  const filter = useSceneStore((s) => s.loopFilter);
  const showRejected = useSceneStore((s) => s.layers.rejected);
  const hover = useSceneStore((s) => s.hover);
  const select = useSceneStore((s) => s.select);
  const raycaster = useThree((s) => s.raycaster);
  useEffect(() => { raycaster.params.Line2 = { threshold: nodeSize * 0.5 }; }, [raycaster, nodeSize]);
  const groups = useMemo(() => loopGroups(scene.loopFlags, filter, showRejected), [scene, filter, showRejected]);
  const crosses = useMemo(() => crossSegments(midpoints(positions, pairsOf(scene, groups.overturned)), nodeSize * 0.8), [scene, positions, groups.overturned, nodeSize]);
  const onHover = (i: number | null) => hover(i === null ? null : { kind: "loop", index: i });
  const onPick = (i: number) => select({ kind: "loop", index: i });
  return (
    <>
      <LoopGroup {...props} indices={groups.accepted} width={2.5} dashed={false} onHover={onHover} onPick={onPick} />
      <LoopGroup {...props} indices={groups.rejected} width={1.5} dashed onHover={onHover} onPick={onPick} />
      <RawSegments segments={crosses} color={palette.reject} opacity={1} />
    </>
  );
}
