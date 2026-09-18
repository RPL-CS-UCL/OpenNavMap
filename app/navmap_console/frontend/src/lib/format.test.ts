import { describe, expect, it } from "vitest";
import { formatBytes, formatDuration } from "./format";

describe("formatBytes", () => {
  it("uses binary units with one decimal", () => {
    expect(formatBytes(0)).toBe("0 B");
    expect(formatBytes(1536)).toBe("1.5 KiB");
    expect(formatBytes(801 * 1024 * 1024)).toBe("801.0 MiB");
  });
});

describe("formatDuration", () => {
  it("formats seconds as h:mm:ss", () => {
    expect(formatDuration(5)).toBe("0:00:05");
    expect(formatDuration(3661)).toBe("1:01:01");
  });
});
