import { describe, expect, it } from "vitest";
import { LOOP_ACCEPTED, LOOP_HIST, LOOP_OVERTURNED, LOOP_REJECTED_NEW, NODE_CULLED } from "@/api/scene-bundle";
import { autoNodeStyle, drawRows, loopGroups, nodeRowIndex, nodeSizeFor } from "./visibility";

describe("visibility", () => {
  it("drops culled rows unless the culled layer is on", () => {
    const flags = new Uint8Array([0, NODE_CULLED, 0]);
    expect(Array.from(drawRows(flags, false))).toEqual([0, 2]);
    expect(Array.from(drawRows(flags, true))).toEqual([0, 1, 2]);
  });
  it("groups loops by state and filter", () => {
    const flags = new Uint8Array([LOOP_ACCEPTED, LOOP_ACCEPTED | LOOP_HIST, LOOP_HIST | LOOP_OVERTURNED, LOOP_REJECTED_NEW]);
    const all = loopGroups(flags, "all", true);
    expect(Array.from(all.accepted)).toEqual([0, 1]);
    expect(Array.from(all.rejected)).toEqual([2, 3]);
    expect(Array.from(all.overturned)).toEqual([2]);
    expect(Array.from(loopGroups(flags, "all", false).rejected)).toEqual([]);
    expect(Array.from(loopGroups(flags, "hist", true).accepted)).toEqual([1]);
    expect(Array.from(loopGroups(flags, "new", true).rejected)).toEqual([3]);
    expect(Array.from(loopGroups(flags, "accepted", true).rejected)).toEqual([]);
  });
  it("auto node style and size", () => {
    expect(autoNodeStyle(100, "auto")).toBe("frustum");
    expect(autoNodeStyle(9000, "auto")).toBe("points");
    expect(autoNodeStyle(9000, "frustum")).toBe("frustum");
    expect(nodeSizeFor(1, 100)).toBeCloseTo(0.35, 5);
    expect(nodeSizeFor(0, 100)).toBeCloseTo(2, 5);
    expect(nodeSizeFor(100, 100)).toBe(2);
  });
  it("maps global node ids to rows", () => {
    expect(nodeRowIndex(new Uint32Array([5, 9])).get(9)).toBe(1);
  });
});
