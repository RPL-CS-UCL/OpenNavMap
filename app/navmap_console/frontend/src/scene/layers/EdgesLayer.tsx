import { useThree } from "@react-three/fiber";
import { useEffect, useMemo } from "react";
import { BufferAttribute, BufferGeometry, LineBasicMaterial } from "three";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
import { LineSegments2 } from "three/examples/jsm/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";
import type { RGB } from "../build/colors";
import { buildSegments } from "../build/segments";

export const noRaycast = () => undefined;

/** Plain 1 px segments from a raw xyz-pair array; not pickable. */
export function RawSegments({ segments, color, opacity }: { segments: Float32Array; color: RGB; opacity: number }) {
  const geometry = useMemo(() => {
    const g = new BufferGeometry();
    g.setAttribute("position", new BufferAttribute(segments, 3));
    return g;
  }, [segments]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  const material = useMemo(() => new LineBasicMaterial({ transparent: true }), []);
  useEffect(() => () => material.dispose(), [material]);
  useEffect(() => {
    material.color.setRGB(color[0], color[1], color[2]);
    material.opacity = opacity;
  }, [material, color, opacity]);
  if (segments.length === 0) return null;
  return <lineSegments geometry={geometry} material={material} frustumCulled={false} raycast={noRaycast} />;
}

/** Screen-space width of the graph edges (odom / covis / trav); WebGL's plain lines are stuck at 1 px. */
const EDGE_WIDTH_PX = 2.5;

export function EdgesLayer({ positions, pairs, color, opacity }: { positions: Float32Array; pairs: Uint32Array; color: RGB; opacity: number }) {
  const size = useThree((s) => s.size);
  const material = useMemo(() => new LineMaterial({ linewidth: EDGE_WIDTH_PX, transparent: true }), []);
  useEffect(() => () => material.dispose(), [material]);
  useEffect(() => { material.resolution.set(size.width, size.height); }, [material, size.width, size.height]);
  useEffect(() => {
    material.color.setRGB(color[0], color[1], color[2]);
    material.opacity = opacity;
  }, [material, color, opacity]);
  const line = useMemo(() => {
    if (pairs.length === 0) return null;
    const geom = new LineSegmentsGeometry();
    geom.setPositions(buildSegments(positions, pairs));
    const obj = new LineSegments2(geom, material);
    obj.frustumCulled = false;
    obj.raycast = noRaycast;
    return obj;
  }, [positions, pairs, material]);
  useEffect(() => () => line?.geometry.dispose(), [line]);
  if (!line) return null;
  return <primitive object={line} />;
}
