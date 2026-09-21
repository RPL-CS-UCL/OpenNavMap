import { Grid } from "@react-three/drei";
import { DoubleSide } from "three";
import { useSceneStore } from "@/stores/scene-store";
import type { Bounds } from "../build/bounds";

export const GRID_CELL_M = 10;
export const GRID_SECTION_M = 50;

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
      <axesHelper args={[GRID_CELL_M]} />
    </>
  );
}
