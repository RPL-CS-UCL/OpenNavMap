import { useEffect, useMemo } from "react";
import { BufferAttribute, BufferGeometry, LineBasicMaterial } from "three";
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

export function EdgesLayer({ positions, pairs, color, opacity }: { positions: Float32Array; pairs: Uint32Array; color: RGB; opacity: number }) {
  const segments = useMemo(() => buildSegments(positions, pairs), [positions, pairs]);
  return <RawSegments segments={segments} color={color} opacity={opacity} />;
}
