export interface Bounds { min: [number, number, number]; max: [number, number, number]; center: [number, number, number]; radius: number }

export function computeBounds(pos: Float32Array, count = pos.length / 3): Bounds {
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < count; i++) {
    for (let k = 0; k < 3; k++) {
      const v = pos[i * 3 + k];
      if (v < min[k]) min[k] = v;
      if (v > max[k]) max[k] = v;
    }
  }
  if (count === 0) return { min: [0, 0, 0], max: [0, 0, 0], center: [0, 0, 0], radius: 1 };
  const center: [number, number, number] = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
  const radius = Math.max(1e-3, Math.hypot(max[0] - center[0], max[1] - center[1], max[2] - center[2]));
  return { min, max, center, radius };
}

/** Camera distance so a sphere of `radius` fits the narrower field of view, with 10 % margin. */
export function fitDistance(radius: number, fovDeg: number, aspect: number): number {
  const v = (fovDeg * Math.PI) / 180;
  const h = 2 * Math.atan(Math.tan(v / 2) * aspect);
  return (radius / Math.sin(Math.min(v, h) / 2)) * 1.1;
}

export function medianEdgeLength(pos: Float32Array, pairs: Uint32Array): number {
  const n = pairs.length / 2;
  if (n === 0) return 0;
  const lens = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const a = pairs[i * 2] * 3, b = pairs[i * 2 + 1] * 3;
    lens[i] = Math.hypot(pos[a] - pos[b], pos[a + 1] - pos[b + 1], pos[a + 2] - pos[b + 2]);
  }
  lens.sort();
  return lens[Math.floor(n / 2)];
}
