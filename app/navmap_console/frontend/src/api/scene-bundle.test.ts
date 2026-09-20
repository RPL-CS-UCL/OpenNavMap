import { describe, expect, it } from "vitest";
import { LOOP_ACCEPTED, NODE_NEW, decodeSceneBundle, encodeSceneBundle, toScene } from "./scene-bundle";
import { makeSceneFixture } from "@/test/scene-fixture";

describe("scene bundle", () => {
  it("round-trips typed arrays with shapes, odd byte lengths and empty arrays", () => {
    const buf = encodeSceneBundle({
      node_id: { data: new Uint32Array([0, 1, 2]), shape: [3] },
      node_pos: { data: new Float32Array([0, 0, 0, 1, 0, 0, 2, 0, 0]), shape: [3, 3] },
      node_flags: { data: new Uint8Array([1, 0, 1]), shape: [3] },
      node_step: { data: new Uint16Array([0, 0, 1]), shape: [3] },
      edge_odom: { data: new Uint32Array(0), shape: [0, 2] },
    });
    expect(new TextDecoder().decode(new Uint8Array(buf, 0, 4))).toBe("NMSB");
    const d = decodeSceneBundle(buf);
    expect(d.shapes.node_pos).toEqual([3, 3]);
    expect(Array.from(d.arrays.node_pos as Float32Array)).toEqual([0, 0, 0, 1, 0, 0, 2, 0, 0]);
    expect(d.arrays.node_flags).toBeInstanceOf(Uint8Array);
    expect(d.shapes.edge_odom).toEqual([0, 2]);
    expect((d.arrays.edge_odom as Uint32Array).length).toBe(0);
  });

  it("rejects a foreign magic and unknown dtype codes", () => {
    expect(() => decodeSceneBundle(new Uint8Array([1, 2, 3, 4, 0, 0, 0, 0]).buffer)).toThrow(/NMSB/);
  });

  it("toScene exposes counts and flat arrays from the fixture", () => {
    const s = toScene(decodeSceneBundle(makeSceneFixture(24, 3)));
    expect(s.numNodes).toBe(24);
    expect(s.numLoops).toBe(3);
    expect(s.pos.length).toBe(72);
    expect(s.odom.length).toBe(23 * 2);
    expect(s.step[0]).toBe(0);
    expect(s.step[23]).toBe(1);
    expect(s.flags[23] & NODE_NEW).toBe(NODE_NEW);
    expect(s.loopFlags[0] & LOOP_ACCEPTED).toBe(LOOP_ACCEPTED);
    expect(s.loopIdx[0]).toBeLessThan(12);
    expect(s.loopIdx[1]).toBeGreaterThanOrEqual(12);
  });
});
