import { LOOP_ACCEPTED, LOOP_HIST, LOOP_OVERTURNED, NODE_CULLED } from "@/api/scene-bundle";
import type { LoopFilter, NodeStyle } from "@/stores/scene-store";

export const POINTS_THRESHOLD = 8000;

export function drawRows(flags: Uint8Array, showCulled: boolean): Uint32Array {
  const rows: number[] = [];
  for (let i = 0; i < flags.length; i++) if (showCulled || !(flags[i] & NODE_CULLED)) rows.push(i);
  return Uint32Array.from(rows);
}

export function loopGroups(loopFlags: Uint8Array, filter: LoopFilter, showRejected: boolean) {
  const accepted: number[] = [], rejected: number[] = [], overturned: number[] = [];
  for (let i = 0; i < loopFlags.length; i++) {
    const f = loopFlags[i];
    const isAccepted = (f & LOOP_ACCEPTED) !== 0;
    const isHist = (f & LOOP_HIST) !== 0;
    if (filter === "accepted" && !isAccepted) continue;
    if (filter === "rejected" && isAccepted) continue;
    if (filter === "hist" && !isHist) continue;
    if (filter === "new" && isHist) continue;
    if (isAccepted) accepted.push(i);
    else if (showRejected) {
      rejected.push(i);
      if (f & LOOP_OVERTURNED) overturned.push(i);
    }
  }
  return { accepted: Uint32Array.from(accepted), rejected: Uint32Array.from(rejected), overturned: Uint32Array.from(overturned) };
}

export function autoNodeStyle(numNodes: number, style: NodeStyle): "frustum" | "points" {
  if (style !== "auto") return style;
  return numNodes > POINTS_THRESHOLD ? "points" : "frustum";
}

/** Frustum size from the odometry spacing; falls back to the scene radius when there are no edges. */
export function nodeSizeFor(medianEdge: number, radius: number): number {
  const base = medianEdge > 0 ? medianEdge * 0.35 : radius / 50;
  return Math.min(2, Math.max(0.02, base));
}

/** Global node id -> row index in the scene arrays. */
export function nodeRowIndex(nodeId: Uint32Array): Map<number, number> {
  const m = new Map<number, number>();
  nodeId.forEach((id, row) => m.set(id, row));
  return m;
}
