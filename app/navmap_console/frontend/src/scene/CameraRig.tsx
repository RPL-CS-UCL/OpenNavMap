import { OrbitControls, OrthographicCamera, PerspectiveCamera } from "@react-three/drei";
import { useFrame, useThree } from "@react-three/fiber";
import { useEffect, useRef } from "react";
import { OrthographicCamera as OrthoImpl, PerspectiveCamera as PerspImpl, Vector3 } from "three";
import { useSceneStore } from "@/stores/scene-store";
import { type Bounds, fitDistance } from "./build/bounds";

interface Controls { target: Vector3; update: () => void }
const FLY_MS = 300;

export function CameraRig({ bounds, ready }: { bounds: Bounds; ready: boolean }) {
  const camera = useThree((s) => s.camera);
  const controls = useThree((s) => s.controls) as unknown as Controls | null;
  const size = useThree((s) => s.size);
  const mode = useSceneStore((s) => s.camera);
  const up = useSceneStore((s) => s.up);
  const fitSeq = useSceneStore((s) => s.fitSeq);
  const flyTo = useSceneStore((s) => s.flyTo);
  const boundsRef = useRef(bounds);
  boundsRef.current = bounds;
  const fly = useRef<{ from: Vector3; to: Vector3; start: number } | null>(null);

  useEffect(() => {
    if (!controls || !ready) return;
    const b = boundsRef.current;
    const center = new Vector3(...b.center);
    const upVec = up === "z" ? new Vector3(0, 0, 1) : new Vector3(0, 1, 0);
    if (mode === "top") {
      camera.up.copy(up === "z" ? new Vector3(0, 1, 0) : new Vector3(0, 0, -1));
      camera.position.copy(center).addScaledVector(upVec, b.radius * 4);
      if (camera instanceof OrthoImpl) {
        camera.zoom = Math.min(size.width, size.height) / (2 * b.radius * 1.1);
        camera.updateProjectionMatrix();
      }
    } else {
      camera.up.copy(upVec);
      const fov = camera instanceof PerspImpl ? camera.fov : 50;
      const d = fitDistance(b.radius, fov, size.width / Math.max(1, size.height));
      const dir = up === "z" ? new Vector3(0.6, -0.6, 0.55) : new Vector3(0.6, 0.55, 0.6);
      camera.position.copy(center).addScaledVector(dir.normalize(), d);
    }
    camera.lookAt(center);
    controls.target.copy(center);
    controls.update();
  }, [controls, ready, fitSeq, mode, up, camera, size.width, size.height]);

  useEffect(() => {
    if (!flyTo || !controls) return;
    fly.current = { from: controls.target.clone(), to: new Vector3(...flyTo.target), start: performance.now() };
  }, [flyTo, controls]);

  useFrame(() => {
    const f = fly.current;
    if (!f || !controls) return;
    const t = Math.min(1, (performance.now() - f.start) / FLY_MS);
    const e = 1 - (1 - t) * (1 - t);
    const next = f.from.clone().lerp(f.to, e);
    camera.position.add(next.clone().sub(controls.target));
    controls.target.copy(next);
    controls.update();
    if (t >= 1) fly.current = null;
  });

  return (
    <>
      {mode === "top"
        ? <OrthographicCamera makeDefault near={-1e4} far={1e4} />
        : <PerspectiveCamera makeDefault fov={50} near={0.01} far={1e4} />}
      <OrbitControls makeDefault enableRotate={mode !== "top"} enableDamping={false} />
    </>
  );
}
