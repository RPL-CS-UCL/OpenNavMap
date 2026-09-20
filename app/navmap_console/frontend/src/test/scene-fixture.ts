import { LOOP_ACCEPTED, LOOP_REJECTED_NEW, NODE_NEW, encodeSceneBundle } from "@/api/scene-bundle";

/** Two straight 12-node runs (step 0 at y=0, step 1 at y=2), an odometry chain with one bridge edge, and `loops` loop factors. */
export function makeSceneFixture(n = 24, loops = 3): ArrayBuffer {
  const half = Math.floor(n / 2);
  const pos = new Float32Array(n * 3);
  const posPre = new Float32Array(n * 3);
  const quat = new Float32Array(n * 4);
  const step = new Uint16Array(n);
  const flags = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const s = i < half ? 0 : 1;
    pos[i * 3] = (i % half) * 0.5;
    pos[i * 3 + 1] = s * 2;
    posPre[i * 3] = pos[i * 3] + (s ? 0.02 : 0);
    posPre[i * 3 + 1] = pos[i * 3 + 1];
    quat[i * 4 + 3] = 1;
    step[i] = s;
    if (s) flags[i] |= NODE_NEW;
  }
  const odom = new Uint32Array((n - 1) * 2);
  for (let i = 0; i < n - 1; i++) {
    odom[i * 2] = i;
    odom[i * 2 + 1] = i + 1;
  }
  const loopIdx = new Uint32Array(loops * 2);
  const loopFlags = new Uint8Array(loops);
  const loopWeight = new Float32Array(loops);
  const loopConf = new Float32Array(loops);
  for (let l = 0; l < loops; l++) {
    loopIdx[l * 2] = l;
    loopIdx[l * 2 + 1] = half + l;
    loopFlags[l] = l === 0 ? LOOP_ACCEPTED : LOOP_REJECTED_NEW;
    loopWeight[l] = l === 0 ? 0.9 : 0.1;
    loopConf[l] = 0.5;
  }
  const nan = new Float32Array(loops).fill(NaN);
  return encodeSceneBundle({
    node_id: { data: Uint32Array.from({ length: n }, (_, i) => i), shape: [n] },
    node_pos: { data: pos, shape: [n, 3] },
    node_pos_pre: { data: posPre, shape: [n, 3] },
    node_quat: { data: quat, shape: [n, 4] },
    node_step: { data: step, shape: [n] },
    node_comp: { data: new Uint16Array(n), shape: [n] },
    node_flags: { data: flags, shape: [n] },
    edge_odom: { data: odom, shape: [n - 1, 2] },
    edge_odom_w: { data: new Float32Array(n - 1).fill(1), shape: [n - 1] },
    edge_covis: { data: new Uint32Array(0), shape: [0, 2] },
    edge_covis_w: { data: new Float32Array(0), shape: [0] },
    edge_trav: { data: new Uint32Array(0), shape: [0, 2] },
    edge_trav_w: { data: new Float32Array(0), shape: [0] },
    loop_idx: { data: loopIdx, shape: [loops, 2] },
    loop_conf: { data: loopConf, shape: [loops] },
    loop_weight: { data: loopWeight, shape: [loops] },
    loop_flags: { data: loopFlags, shape: [loops] },
    loop_terr: { data: nan, shape: [loops] },
    loop_rerr: { data: nan, shape: [loops] },
  });
}
