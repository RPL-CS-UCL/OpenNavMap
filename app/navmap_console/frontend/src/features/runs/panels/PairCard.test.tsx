import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { PairCard } from "./PairCard";

describe("PairCard", () => {
  it("shows the pair image and the loop metrics", () => {
    render(<PairCard rid="reg_1" runId="run_20260918_120000_cd34" a={3} b={13} meta={{ weight: 0.9, conf: 0.8, terr: null, rerr: null }} open onOpenChange={() => {}} />);
    expect(screen.getByText("Loop pair 3 · 13")).toBeInTheDocument();
    expect(screen.getByRole("img")).toHaveAttribute("src", expect.stringContaining("/pairs/3/13/image"));
    expect(screen.getByText(/0\.9/)).toBeInTheDocument();
  });
});
