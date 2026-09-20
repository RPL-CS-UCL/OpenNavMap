// Binary contract shared with backend/navmap_console/readers/scene_bundle.py (NMSB v1).
// Layout: 'NMSB' | u32 version | u32 count | per array: u16 nameLen | name | u8 dtype | u8 ndim | u32 shape[ndim] | pad4 | data | pad4
export type TypedArray = Uint8Array | Uint16Array | Uint32Array | Float32Array;
export interface DecodedBundle {
  arrays: Record<string, TypedArray>;
  shapes: Record<string, number[]>;
}

const MAGIC = 0x42534d4e; // "NMSB" little-endian
const VERSION = 1;
const CTORS = [Uint8Array, Uint16Array, Uint32Array, Float32Array] as const;
const pad4 = (n: number): number => (4 - (n % 4)) % 4;

function dtypeCode(a: TypedArray): number {
  const i = CTORS.findIndex((c) => a instanceof c);
  if (i < 0) throw new TypeError("unsupported typed array");
  return i;
}

export function encodeSceneBundle(arrays: Record<string, { data: TypedArray; shape: number[] }>): ArrayBuffer {
  const enc = new TextEncoder();
  const entries = Object.entries(arrays).map(([name, { data, shape }]) => ({ name: enc.encode(name), data, shape }));
  let size = 12;
  for (const e of entries) {
    size += 2 + e.name.length + 2 + 4 * e.shape.length;
    size += pad4(size) + e.data.byteLength;
    size += pad4(size);
  }
  const buf = new ArrayBuffer(size);
  const view = new DataView(buf);
  const bytes = new Uint8Array(buf);
  view.setUint32(0, MAGIC, true);
  view.setUint32(4, VERSION, true);
  view.setUint32(8, entries.length, true);
  let off = 12;
  for (const e of entries) {
    view.setUint16(off, e.name.length, true);
    off += 2;
    bytes.set(e.name, off);
    off += e.name.length;
    view.setUint8(off, dtypeCode(e.data));
    view.setUint8(off + 1, e.shape.length);
    off += 2;
    for (const s of e.shape) {
      view.setUint32(off, s, true);
      off += 4;
    }
    off += pad4(off);
    bytes.set(new Uint8Array(e.data.buffer, e.data.byteOffset, e.data.byteLength), off);
    off += e.data.byteLength;
    off += pad4(off);
  }
  return buf;
}

export function decodeSceneBundle(buf: ArrayBuffer): DecodedBundle {
  const view = new DataView(buf);
  if (buf.byteLength < 12 || view.getUint32(0, true) !== MAGIC) throw new Error("not an NMSB scene bundle");
  const version = view.getUint32(4, true);
  if (version !== VERSION) throw new Error(`unsupported NMSB version ${version}`);
  const count = view.getUint32(8, true);
  const dec = new TextDecoder();
  const arrays: Record<string, TypedArray> = {};
  const shapes: Record<string, number[]> = {};
  let off = 12;
  for (let i = 0; i < count; i++) {
    const nameLen = view.getUint16(off, true);
    off += 2;
    const name = dec.decode(new Uint8Array(buf, off, nameLen));
    off += nameLen;
    const code = view.getUint8(off);
    const ndim = view.getUint8(off + 1);
    off += 2;
    const shape: number[] = [];
    for (let d = 0; d < ndim; d++) {
      shape.push(view.getUint32(off, true));
      off += 4;
    }
    off += pad4(off);
    const Ctor = CTORS[code];
    if (!Ctor) throw new Error(`unknown dtype code ${code} for ${name}`);
    const length = shape.reduce((a, b) => a * b, 1);
    // Copy out of the shared buffer so each array owns aligned memory (WebGL uploads want that).
    arrays[name] = new Ctor(buf.slice(off, off + length * Ctor.BYTES_PER_ELEMENT)) as TypedArray;
    shapes[name] = shape;
    off += length * Ctor.BYTES_PER_ELEMENT;
    off += pad4(off);
  }
  return { arrays, shapes };
}

export const NODE_NEW = 1;
export const NODE_CULLED = 2;
export const NODE_NOT_COVIS = 4;
export const LOOP_ACCEPTED = 1;
export const LOOP_HIST = 2;
export const LOOP_OVERTURNED = 4;
export const LOOP_REJECTED_NEW = 8;

export interface Scene {
  numNodes: number;
  numLoops: number;
  nodeId: Uint32Array;
  pos: Float32Array;
  posPre: Float32Array;
  quat: Float32Array;
  step: Uint16Array;
  comp: Uint16Array;
  flags: Uint8Array;
  odom: Uint32Array;
  odomW: Float32Array;
  covis: Uint32Array;
  covisW: Float32Array;
  trav: Uint32Array;
  travW: Float32Array;
  loopIdx: Uint32Array;
  loopConf: Float32Array;
  loopWeight: Float32Array;
  loopFlags: Uint8Array;
  loopTerr: Float32Array;
  loopRerr: Float32Array;
}

function pick<T extends TypedArray>(d: DecodedBundle, name: string, Ctor: new (n: number) => T): T {
  const a = d.arrays[name];
  return a instanceof Ctor ? a : new Ctor(0);
}

export function toScene(d: DecodedBundle): Scene {
  const nodeId = pick(d, "node_id", Uint32Array);
  const loopIdx = pick(d, "loop_idx", Uint32Array);
  return {
    numNodes: nodeId.length,
    numLoops: loopIdx.length / 2,
    nodeId,
    pos: pick(d, "node_pos", Float32Array),
    posPre: pick(d, "node_pos_pre", Float32Array),
    quat: pick(d, "node_quat", Float32Array),
    step: pick(d, "node_step", Uint16Array),
    comp: pick(d, "node_comp", Uint16Array),
    flags: pick(d, "node_flags", Uint8Array),
    odom: pick(d, "edge_odom", Uint32Array),
    odomW: pick(d, "edge_odom_w", Float32Array),
    covis: pick(d, "edge_covis", Uint32Array),
    covisW: pick(d, "edge_covis_w", Float32Array),
    trav: pick(d, "edge_trav", Uint32Array),
    travW: pick(d, "edge_trav_w", Float32Array),
    loopIdx,
    loopConf: pick(d, "loop_conf", Float32Array),
    loopWeight: pick(d, "loop_weight", Float32Array),
    loopFlags: pick(d, "loop_flags", Uint8Array),
    loopTerr: pick(d, "loop_terr", Float32Array),
    loopRerr: pick(d, "loop_rerr", Float32Array),
  };
}
