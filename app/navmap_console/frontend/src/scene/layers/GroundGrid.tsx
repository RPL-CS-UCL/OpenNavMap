import { Grid } from "@react-three/drei";
import { useThree } from "@react-three/fiber";
import { useEffect, useMemo } from "react";
import { DoubleSide } from "three";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
import { LineSegments2 } from "three/examples/jsm/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";
import { useSceneStore } from "@/stores/scene-store";
import type { Bounds } from "../build/bounds";

export const GRID_CELL_M = 10;
export const GRID_SECTION_M = 50;
/** Screen-space width of the origin axes; three's axesHelper is a 1 px WebGL line and cannot be thickened. */
const AXES_WIDTH_PX = 7;

/** X red, Y green, Z blue from the world origin (frame 0), drawn as thick screen-space lines. */
function OriginAxes({ length }: { length: number }) {
  const size = useThree((s) => s.size);
  const material = useMemo(() => new LineMaterial({ vertexColors: true, linewidth: AXES_WIDTH_PX }), []);
  useEffect(() => () => material.dispose(), [material]);
  useEffect(() => { material.resolution.set(size.width, size.height); }, [material, size.width, size.height]);
  const line = useMemo(() => {
    const geom = new LineSegmentsGeometry();
    geom.setPositions([0, 0, 0, length, 0, 0, 0, 0, 0, 0, length, 0, 0, 0, 0, 0, 0, length]);
    geom.setColors([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]);
    const obj = new LineSegments2(geom, material);
    obj.frustumCulled = false;
    obj.raycast = () => undefined;
    return obj;
  }, [length, material]);
  useEffect(() => () => line.geometry.dispose(), [line]);
  return <primitive object={line} />;
}

/** Metric ground grid just below the lowest node (lines through world x=0 / y=0) and the axes of frame 0. */
export function GroundGrid({ bounds }: { bounds: Bounds }) {
  const up = useSceneStore((s) => s.up);
  const floor = bounds.min[up === "z" ? 2 : 1] - Math.max(0.5, bounds.radius * 0.02);
  const position: [number, number, number] = up === "z" ? [0, 0, floor] : [0, floor, 0];
  // drei's Grid lies in its local XZ plane (normal +Y); tilt it so the normal is world Z when Z is up
  const rotation: [number, number, number] = up === "z" ? [Math.PI / 2, 0, 0] : [0, 0, 0];
  return (
    <>
      <Grid
        position={position}
        rotation={rotation}
        cellSize={GRID_CELL_M}
        sectionSize={GRID_SECTION_M}
        cellColor="#9ca3af"
        sectionColor="#6b7280"
        cellThickness={0.6}
        sectionThickness={1.2}
        infiniteGrid
        fadeDistance={bounds.radius * 8 + 100}
        side={DoubleSide}
      />
      <OriginAxes length={GRID_CELL_M} />
    </>
  );
}
