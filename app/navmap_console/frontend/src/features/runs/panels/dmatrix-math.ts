import type { Scene } from "@/api/scene-bundle";
import type { DMatrixCandidate } from "@/api/types";

/** Pixel size of one matrix cell at zoom 1. */
export const MATRIX_CELL_PX = 16;

export interface PanZoom { zoom: number; pan: { x: number; y: number } }

/** Multiply the zoom around `cursor` so that pixel stays put; zoom is clamped to 1..32. */
export function zoomAtPoint(zoom: number, pan: { x: number; y: number }, cursor: { x: number; y: number }, factor: number): PanZoom {
  const z = Math.min(32, Math.max(1, zoom * factor));
  const k = z / zoom;
  return { zoom: z, pan: { x: cursor.x - (cursor.x - pan.x) * k, y: cursor.y - (cursor.y - pan.y) * k } };
}

/** Pixel coordinates (relative to the unzoomed image) -> matrix cell. */
export function cellAtPoint(x: number, y: number, zoom: number, pan: { x: number; y: number }, cell = MATRIX_CELL_PX) {
  return {
    col: Math.floor((x - pan.x) / zoom / cell),
    row: Math.floor((y - pan.y) / zoom / cell),
  };
}

/** The candidate whose cell is nearest to (col, row), if any lies within `radius` cells. */
export function nearestCandidate(col: number, row: number, candidates: DMatrixCandidate[],
                                 rowOf: Map<number, number>, colOf: Map<number, number>, radius: number) {
  let best: DMatrixCandidate | null = null;
  let bestDist = Infinity;
  for (const c of candidates) {
    const cc = colOf.get(c.query);
    const rr = rowOf.get(c.db);
    if (cc === undefined || rr === undefined) continue;
    const d = Math.hypot(cc - col, rr - row);
    if (d <= radius && d < bestDist) {
      best = c;
      bestDist = d;
    }
  }
  return best;
}

/** Row of the loop factor (a, b) in either orientation; -1 when absent. */
export function findLoopIndex(scene: Scene | null, a: number, b: number): number {
  if (!scene) return -1;
  for (let i = 0; i < scene.numLoops; i++) {
    const x = scene.loopIdx[i * 2];
    const y = scene.loopIdx[i * 2 + 1];
    if ((x === a && y === b) || (x === b && y === a)) return i;
  }
  return -1;
}
