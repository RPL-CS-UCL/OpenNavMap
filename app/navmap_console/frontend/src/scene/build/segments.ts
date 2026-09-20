/** Pure array builders for the 3D scene; no three.js imports so they run in node tests. */

export function buildSegments(pos: Float32Array, pairs: Uint32Array, out?: Float32Array): Float32Array {
  const n = pairs.length / 2;
  const seg = out && out.length >= n * 6 ? out : new Float32Array(n * 6);
  for (let i = 0; i < n; i++) {
    const a = pairs[i * 2] * 3;
    const b = pairs[i * 2 + 1] * 3;
    const o = i * 6;
    seg[o] = pos[a]; seg[o + 1] = pos[a + 1]; seg[o + 2] = pos[a + 2];
    seg[o + 3] = pos[b]; seg[o + 4] = pos[b + 1]; seg[o + 5] = pos[b + 2];
  }
  return seg;
}

export function midpoints(pos: Float32Array, pairs: Uint32Array): Float32Array {
  const n = pairs.length / 2;
  const out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const a = pairs[i * 2] * 3;
    const b = pairs[i * 2 + 1] * 3;
    for (let k = 0; k < 3; k++) out[i * 3 + k] = (pos[a + k] + pos[b + k]) / 2;
  }
  return out;
}

/** Three axis-aligned strokes through each point: the "x" marker for overturned loop edges. */
export function crossSegments(points: Float32Array, size: number): Float32Array {
  const n = points.length / 3;
  const out = new Float32Array(n * 18);
  const h = size / 2;
  for (let i = 0; i < n; i++) {
    const x = points[i * 3], y = points[i * 3 + 1], z = points[i * 3 + 2];
    out.set([x - h, y, z, x + h, y, z, x, y - h, z, x, y + h, z, x, y, z - h, x, y, z + h], i * 18);
  }
  return out;
}

export function lerpPositions(a: Float32Array, b: Float32Array, t: number, out?: Float32Array): Float32Array {
  const res = out && out.length >= a.length ? out : new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) res[i] = a[i] + (b[i] - a[i]) * t;
  return res;
}

/** Segments pre -> post for nodes PGO moved by more than `threshold` metres. */
export function buildDisplacements(pos: Float32Array, posPre: Float32Array, threshold: number): { segments: Float32Array; moved: Uint32Array } {
  const n = Math.min(pos.length, posPre.length) / 3;
  const idx: number[] = [];
  const t2 = threshold * threshold;
  for (let i = 0; i < n; i++) {
    const dx = pos[i * 3] - posPre[i * 3], dy = pos[i * 3 + 1] - posPre[i * 3 + 1], dz = pos[i * 3 + 2] - posPre[i * 3 + 2];
    if (dx * dx + dy * dy + dz * dz > t2) idx.push(i);
  }
  const segments = new Float32Array(idx.length * 6);
  idx.forEach((i, j) => {
    segments.set([posPre[i * 3], posPre[i * 3 + 1], posPre[i * 3 + 2], pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]], j * 6);
  });
  return { segments, moved: Uint32Array.from(idx) };
}
