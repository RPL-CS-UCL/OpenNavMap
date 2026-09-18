import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { LogConsole } from "./LogConsole";

const ESC = String.fromCharCode(27);

describe("LogConsole", () => {
  it("renders lines with ANSI colours turned into spans and filters by search", async () => {
    const lines = ["plain line", `${ESC}[31mred error${ESC}[0m`, "PGO: final error: 0.456"];
    render(<LogConsole lines={lines} initialItemCount={3} />);
    expect(screen.getByText("plain line")).toBeInTheDocument();
    expect(screen.getByText("red error")).toBeInTheDocument();
    await userEvent.type(screen.getByRole("searchbox"), "PGO");
    expect(screen.queryByText("plain line")).not.toBeInTheDocument();
    expect(screen.getByText(/PGO: final error/)).toBeInTheDocument();
    expect(screen.getByText("1 / 3")).toBeInTheDocument();
  });
});
