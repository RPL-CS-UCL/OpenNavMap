import { type ThreeEvent, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import { BufferAttribute, BufferGeometry, InstancedBufferAttribute, InstancedMesh, MeshLambertMaterial, PointsMaterial } from "three";
import { NODE_NEW, type Scene } from "@/api/scene-bundle";
import { useSceneStore } from "@/stores/scene-store";
import { type Palette, nodeColors } from "../build/colors";
import { FRUSTUM_INDICES, FRUSTUM_POSITIONS, frustumMatrices } from "../build/instances";
import { drawRows } from "../build/visibility";

interface Props { scene: Scene; positions: Float32Array; palette: Palette; nodeSize: number; style: "frustum" | "points" }
interface Compact { pos: Float32Array; quat: Float32Array; isNew: Uint8Array }
interface Handlers { onPointerMove: (e: ThreeEvent<PointerEvent>) => void; onPointerOut: () => void; onClick: (e: ThreeEvent<MouseEvent>) => void }
const APPEAR_MS = 250;

export function NodesLayer({ scene, positions, palette, nodeSize, style }: Props) {
  const step = useSceneStore((s) => s.step) ?? 0;
  const colorMode = useSceneStore((s) => s.colorMode);
  const pinned = useSceneStore((s) => s.pinned);
  const showCulled = useSceneStore((s) => s.layers.culled);
  const hover = useSceneStore((s) => s.hover);
  const select = useSceneStore((s) => s.select);

  const rows = useMemo(() => drawRows(scene.flags, showCulled), [scene, showCulled]);
  const colors = useMemo(() => {
    const all = nodeColors(scene, colorMode, step, pinned, palette);
    const out = new Float32Array(rows.length * 3);
    rows.forEach((r, i) => out.set(all.subarray(r * 3, r * 3 + 3), i * 3));
    return out;
  }, [scene, colorMode, step, pinned, palette, rows]);
  const compact = useMemo<Compact>(() => {
    const pos = new Float32Array(rows.length * 3), quat = new Float32Array(rows.length * 4), isNew = new Uint8Array(rows.length);
    rows.forEach((r, i) => {
      pos.set(positions.subarray(r * 3, r * 3 + 3), i * 3);
      quat.set(scene.quat.subarray(r * 4, r * 4 + 4), i * 4);
      isNew[i] = scene.flags[r] & NODE_NEW ? 1 : 0;
    });
    return { pos, quat, isNew };
  }, [rows, positions, scene]);

  const lastHover = useRef<number | null>(null);
  const rowOf = (e: ThreeEvent<PointerEvent> | ThreeEvent<MouseEvent>) => e.instanceId ?? e.index;
  const handlers: Handlers = {
    onPointerMove: (e) => {
      const i = rowOf(e);
      if (i === undefined) return;
      e.stopPropagation();
      const id = scene.nodeId[rows[i]];
      if (lastHover.current !== id) { lastHover.current = id; hover({ kind: "node", id }); }
    },
    onPointerOut: () => { if (lastHover.current !== null) { lastHover.current = null; hover(null); } },
    onClick: (e) => {
      const i = rowOf(e);
      if (i === undefined) return;
      e.stopPropagation();
      select({ kind: "node", id: scene.nodeId[rows[i]] });
    },
  };
  return style === "points"
    ? <PointNodes compact={compact} colors={colors} nodeSize={nodeSize} handlers={handlers} />
    : <FrustumNodes compact={compact} colors={colors} nodeSize={nodeSize} sceneKey={scene} handlers={handlers} />;
}

function FrustumNodes({ compact, colors, nodeSize, sceneKey, handlers }: { compact: Compact; colors: Float32Array; nodeSize: number; sceneKey: Scene; handlers: Handlers }) {
  const geometry = useMemo(() => {
    const g = new BufferGeometry();
    g.setAttribute("position", new BufferAttribute(FRUSTUM_POSITIONS, 3));
    g.setIndex(new BufferAttribute(FRUSTUM_INDICES, 1));
    g.computeVertexNormals();
    return g;
  }, []);
  const material = useMemo(() => new MeshLambertMaterial(), []);
  useEffect(() => () => { geometry.dispose(); material.dispose(); }, [geometry, material]);
  const count = compact.pos.length / 3;
  const mesh = useMemo(() => {
    const m = new InstancedMesh(geometry, material, Math.max(1, count));
    m.count = count;
    m.frustumCulled = false;
    return m;
  }, [geometry, material, count]);
  useEffect(() => () => mesh.dispose(), [mesh]);
  const matrices = useMemo(() => frustumMatrices(compact.pos, compact.quat, nodeSize), [compact, nodeSize]);
  useEffect(() => {
    (mesh.instanceMatrix.array as Float32Array).set(matrices);
    mesh.instanceMatrix.needsUpdate = true;
    mesh.instanceColor = new InstancedBufferAttribute(colors, 3);
  }, [mesh, matrices, colors]);

  const appear = useRef({ start: 0, done: true });
  useEffect(() => { appear.current = { start: performance.now(), done: false }; }, [sceneKey]);
  useFrame(() => {
    const a = appear.current;
    if (a.done) return;
    const t = Math.min(1, (performance.now() - a.start) / APPEAR_MS);
    const s = 1 - (1 - t) ** 3;
    const arr = mesh.instanceMatrix.array as Float32Array;
    for (let i = 0; i < count; i++) {
      if (!compact.isNew[i]) continue;
      const o = i * 16;
      for (let k = 0; k < 12; k++) arr[o + k] = matrices[o + k] * s;
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (t >= 1) a.done = true;
  });
  return <primitive object={mesh} {...handlers} />;
}

function PointNodes({ compact, colors, nodeSize, handlers }: { compact: Compact; colors: Float32Array; nodeSize: number; handlers: Handlers }) {
  const raycaster = useThree((s) => s.raycaster);
  useEffect(() => { raycaster.params.Points = { threshold: nodeSize }; }, [raycaster, nodeSize]);
  const geometry = useMemo(() => {
    const g = new BufferGeometry();
    g.setAttribute("position", new BufferAttribute(compact.pos, 3));
    g.setAttribute("color", new BufferAttribute(colors, 3));
    return g;
  }, [compact, colors]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  const material = useMemo(() => new PointsMaterial({ size: 6, sizeAttenuation: false, vertexColors: true }), []);
  useEffect(() => () => material.dispose(), [material]);
  return <points geometry={geometry} material={material} frustumCulled={false} {...handlers} />;
}
