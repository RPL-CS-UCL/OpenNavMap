/** Camera frustum: apex at the optical centre, base 1 unit along +z (camera looks down +z in c2w). */
export const FRUSTUM_POSITIONS = new Float32Array([0, 0, 0, -0.8, -0.5, 1, 0.8, -0.5, 1, 0.8, 0.5, 1, -0.8, 0.5, 1]);
export const FRUSTUM_INDICES = new Uint16Array([0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1, 1, 3, 2, 1, 4, 3]);

/** Column-major 4x4 per node: rotation from xyzw quaternion, uniform scale `size`, translation `pos`. */
export function frustumMatrices(pos: Float32Array, quat: Float32Array, size: number, out?: Float32Array): Float32Array {
  const n = pos.length / 3;
  const m = out && out.length >= n * 16 ? out : new Float32Array(n * 16);
  for (let i = 0; i < n; i++) {
    const x = quat[i * 4], y = quat[i * 4 + 1], z = quat[i * 4 + 2], w = quat[i * 4 + 3];
    const xx = x * x, yy = y * y, zz = z * z, xy = x * y, xz = x * z, yz = y * z, wx = w * x, wy = w * y, wz = w * z;
    const o = i * 16;
    m[o] = (1 - 2 * (yy + zz)) * size; m[o + 1] = 2 * (xy + wz) * size; m[o + 2] = 2 * (xz - wy) * size; m[o + 3] = 0;
    m[o + 4] = 2 * (xy - wz) * size; m[o + 5] = (1 - 2 * (xx + zz)) * size; m[o + 6] = 2 * (yz + wx) * size; m[o + 7] = 0;
    m[o + 8] = 2 * (xz + wy) * size; m[o + 9] = 2 * (yz - wx) * size; m[o + 10] = (1 - 2 * (xx + yy)) * size; m[o + 11] = 0;
    m[o + 12] = pos[i * 3]; m[o + 13] = pos[i * 3 + 1]; m[o + 14] = pos[i * 3 + 2]; m[o + 15] = 1;
  }
  return m;
}
