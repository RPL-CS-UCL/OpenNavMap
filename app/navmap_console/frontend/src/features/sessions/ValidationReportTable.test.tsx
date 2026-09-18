import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { session, sessionInvalid } from "@/test/fixtures";
import { ValidationReportTable } from "./ValidationReportTable";

describe("ValidationReportTable", () => {
  it("renders one row per file with status text, not only color", () => {
    render(<ValidationReportTable report={session.validation!} />);
    expect(screen.getAllByRole("row")).toHaveLength(session.validation!.files.length + 1);
    expect(screen.getByText("poses.txt").closest("tr")).toHaveTextContent("ok");
    expect(screen.getByText("poses_abs_gt.txt").closest("tr")).toHaveTextContent("missing");
  });

  it("lists errors above the table", () => {
    render(<ValidationReportTable report={sessionInvalid.validation!} />);
    expect(screen.getByText(/gps_data.txt is required but missing/)).toBeInTheDocument();
  });
});
