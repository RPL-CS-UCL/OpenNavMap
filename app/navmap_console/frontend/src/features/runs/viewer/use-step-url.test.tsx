import { act, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { MemoryRouter } from "react-router-dom";
import { useLocation } from "react-router-dom";
import { useSceneStore } from "@/stores/scene-store";
import { paramsToSearch, parseViewerParams, useStepUrl } from "./use-step-url";

function Probe() {
  useStepUrl();
  const loc = useLocation();
  return <div data-testid="url">{paramsToSearch(parseViewerParams(new URLSearchParams(loc.search)))}</div>;
}

function Harness({ entry }: { entry: string }) {
  return (
    <MemoryRouter initialEntries={[entry]}>
      <Probe />
    </MemoryRouter>
  );
}

describe("use-step-url", () => {
  beforeEach(() => useSceneStore.getState().reset());

  it("parses step, node and loop edge params, ignoring malformed values", () => {
    const sp = new URLSearchParams("step=3&node=1287&edge=loop%3A57&step=xx");
    expect(parseViewerParams(sp)).toEqual({ step: 3, node: 1287, edge: { kind: "loop", index: 57 } });
    expect(parseViewerParams(new URLSearchParams("step=-1&node=abc&edge=loop:x"))).toEqual({});
    expect(parseViewerParams(new URLSearchParams("edge=node:5"))).toEqual({});
  });

  it("round-trips params through the search string", () => {
    expect(paramsToSearch({ step: 3, node: 1287, edge: { kind: "loop", index: 57 } })).toBe("step=3&node=1287&edge=loop%3A57");
  });

  it("applies URL params to the store on mount and mirrors store changes back", () => {
    render(<Harness entry="/regions/r1/runs/run1?step=1&edge=loop%3A2" />);
    const s = useSceneStore.getState();
    expect(s.step).toBe(1);
    expect(s.follow).toBe(false);
    expect(s.selected).toEqual({ kind: "loop", index: 2 });
    act(() => s.setStep(0));
    expect(screen.getByTestId("url").textContent).toContain("step=0");
    act(() => s.select({ kind: "node", id: 7 }));
    expect(screen.getByTestId("url").textContent).toContain("node=7");
  });
});
