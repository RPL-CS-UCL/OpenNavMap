import { beforeEach, describe, expect, it } from "vitest";
import { useSceneStore } from "./scene-store";

describe("scene store", () => {
  beforeEach(() => useSceneStore.getState().reset());

  it("clamps steps, follow tracks maxStep until the user moves", () => {
    const s = useSceneStore.getState();
    s.setMaxStep(4);
    expect(useSceneStore.getState().step).toBe(4);
    s.stepBy(-2);
    expect(useSceneStore.getState().step).toBe(2);
    expect(useSceneStore.getState().follow).toBe(false);
    s.setMaxStep(6);
    expect(useSceneStore.getState().step).toBe(2);
    s.stepBy(99);
    expect(useSceneStore.getState().step).toBe(6);
    s.jumpToLatest();
    expect(useSceneStore.getState().follow).toBe(true);
  });

  it("keeps at most five pins, oldest dropped", () => {
    const s = useSceneStore.getState();
    for (const k of [1, 2, 3, 4, 5, 6]) s.togglePin(k);
    expect(useSceneStore.getState().pinned).toEqual([2, 3, 4, 5, 6]);
    s.togglePin(3);
    expect(useSceneStore.getState().pinned).toEqual([2, 4, 5, 6]);
  });

  it("layers, camera and fly requests", () => {
    const s = useSceneStore.getState();
    s.toggleLayer("covis");
    expect(useSceneStore.getState().layers.covis).toBe(true);
    s.toggleCamera();
    expect(useSceneStore.getState().camera).toBe("top");
    s.requestFly([1, 2, 3]);
    s.requestFly([4, 5, 6]);
    expect(useSceneStore.getState().flyTo).toEqual({ target: [4, 5, 6], seq: 2 });
    s.select({ kind: "node", id: 7 });
    s.reset();
    expect(useSceneStore.getState().selected).toBeNull();
  });
});
