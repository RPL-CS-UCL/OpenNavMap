import { describe, expect, it } from "vitest";
import { LOOP_ACCEPTED, LOOP_OVERTURNED, NODE_CULLED, NODE_NEW } from "@/api/scene-bundle";
import { computeBounds, fitDistance, medianEdgeLength } from "./bounds";
import { type Palette, loopColor, nodeColors, parseHslTriple, seqColor, srgbToLinear } from "./colors";
import { FRUSTUM_INDICES, FRUSTUM_POSITIONS, frustumMatrices } from "./instances";
import { buildDisplacements, buildSegments, crossSegments, lerpPositions, midpoints } from "./segments";

const rgb = (r: number, g: number, b: number): [number, number, number] => [r, g, b];
const palette: Palette = {
  ref: rgb(0.5, 0.5, 0.5), current: rgb(1, 0.5, 0), accept: rgb(0, 1, 0), reject: rgb(1, 0, 0),
  ghost: rgb(0.7, 0.7, 0.7), culled: rgb(0.2, 0.2, 0.2), select: rgb(1, 1, 0),
  pins: [rgb(0, 0, 1), rgb(0, 1, 1)], seq: [rgb(0.9, 0.9, 1), rgb(0.5, 0.5, 1), rgb(0.1, 0.1, 1)],
  series: [rgb(1, 0, 1), rgb(0, 1, 1), rgb(1, 1, 0)],
};

describe("segments", () => {
  const pos = new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0]);
  it("expands index pairs into xyz pairs and midpoints", () => {
    const seg = buildSegments(pos, new Uint32Array([0, 1, 1, 2]));
    expect(Array.from(seg)).toEqual([0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0]);
    expect(Array.from(midpoints(pos, new Uint32Array([0, 2])))).toEqual([0.5, 0.5, 0]);
    expect(crossSegments(new Float32Array([1, 1, 1]), 2).length).toBe(18);
  });
  it("lerps and finds moved nodes", () => {
    const pre = new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0.5]);
    expect(Array.from(lerpPositions(pre, pos, 0.5))).toEqual([0, 0, 0, 1, 0, 0, 1, 1, 0.25]);
    const d = buildDisplacements(pos, pre, 0.1);
    expect(Array.from(d.moved)).toEqual([2]);
    expect(Array.from(d.segments)).toEqual([1, 1, 0.5, 1, 1, 0]);
  });
});

describe("colors", () => {
  it("parses shadcn HSL triples", () => {
    expect(parseHslTriple("0 0% 100%")).toEqual([1, 1, 1]);
    expect(parseHslTriple("120 100% 50%").map((v) => Math.round(v * 100) / 100)).toEqual([0, 1, 0]);
    expect(srgbToLinear([1, 0, 0.5])[2]).toBeCloseTo(0.214, 3);
  });
  it("focus mode: current step, pins, reference, culled", () => {
    const scene = { numNodes: 4, step: new Uint16Array([0, 1, 2, 2]), comp: new Uint16Array(4), flags: new Uint8Array([0, 0, NODE_NEW, NODE_NEW | NODE_CULLED]) };
    const c = nodeColors(scene, "focus", 2, [1], palette);
    expect(Array.from(c.slice(0, 3))).toEqual(palette.ref);
    expect(Array.from(c.slice(3, 6))).toEqual(palette.pins[0]);
    expect(Array.from(c.slice(6, 9))).toEqual(palette.current);
    expect(Array.from(c.slice(9, 12))).toEqual(palette.culled.map(Math.fround));
  });
  it("order mode is monotonic in step; component mode caps at three series", () => {
    const scene = { numNodes: 3, step: new Uint16Array([0, 1, 2]), comp: new Uint16Array([0, 3, 2]), flags: new Uint8Array(3) };
    const o = nodeColors(scene, "order", 2, [], palette);
    expect(o[0]).toBeGreaterThan(o[3]);
    expect(o[3]).toBeGreaterThan(o[6]);
    expect(seqColor(0.5, palette.seq)).toEqual([0.5, 0.5, 1]);
    const k = nodeColors(scene, "component", 2, [], palette);
    expect(Array.from(k.slice(0, 3))).toEqual(palette.series[0]);
    expect(Array.from(k.slice(3, 6))).toEqual(palette.ghost.map(Math.fround));
    expect(Array.from(k.slice(6, 9))).toEqual(palette.series[2]);
  });
  it("loop color mixes toward ghost as weight drops", () => {
    expect(loopColor(LOOP_ACCEPTED, 1, palette)).toEqual(palette.accept);
    const weak = loopColor(LOOP_ACCEPTED, 0, palette);
    expect(weak[0]).toBeGreaterThan(0);
    expect(loopColor(LOOP_OVERTURNED, 1, palette)).toEqual(palette.reject);
  });
});

describe("bounds and instances", () => {
  it("bounds, fit distance and median edge", () => {
    const pos = new Float32Array([0, 0, 0, 2, 0, 0, 2, 2, 0]);
    const b = computeBounds(pos);
    expect(b.center).toEqual([1, 1, 0]);
    expect(b.radius).toBeCloseTo(Math.SQRT2, 5);
    expect(fitDistance(1, 90, 1)).toBeCloseTo(1.1 * Math.SQRT2, 5);
    expect(medianEdgeLength(pos, new Uint32Array([0, 1, 1, 2, 0, 2]))).toBe(2);
    expect(medianEdgeLength(pos, new Uint32Array(0))).toBe(0);
  });
  it("frustum geometry and matrices", () => {
    expect(FRUSTUM_POSITIONS.length).toBe(15);
    expect(FRUSTUM_INDICES.length).toBe(18);
    const m = frustumMatrices(new Float32Array([1, 2, 3]), new Float32Array([0, 0, Math.SQRT1_2, Math.SQRT1_2]), 2);
    expect(m.length).toBe(16);
    expect([m[12], m[13], m[14], m[15]]).toEqual([1, 2, 3, 1]);
    // 90° about z scaled by 2: x axis -> (0, 2, 0)
    expect(m[0]).toBeCloseTo(0, 6);
    expect(m[1]).toBeCloseTo(2, 6);
  });
});
