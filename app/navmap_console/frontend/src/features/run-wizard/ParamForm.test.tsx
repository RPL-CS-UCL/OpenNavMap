import { useState } from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { paramSpecs } from "@/test/fixtures";
import { ParamForm } from "./ParamForm";
import { buildParamSchema, defaultsOf, diffParams } from "./param-schema";

describe("param-schema", () => {
  it("builds a zod schema from specs and validates types", () => {
    const schema = buildParamSchema(paramSpecs);
    expect(schema.safeParse(defaultsOf(paramSpecs)).success).toBe(true);
    expect(schema.safeParse({ ...defaultsOf(paramSpecs), vpr_match_seq_len: 2.5 }).success).toBe(false);
    expect(schema.safeParse({ ...defaultsOf(paramSpecs), pgo_robust: "bogus" }).success).toBe(false);
  });
  it("lists modified params only", () => {
    expect(diffParams(paramSpecs, { ...defaultsOf(paramSpecs), pgo_loop_sigma_trans: 0.2 })).toEqual([
      { name: "pgo_loop_sigma_trans", from: 0.1, to: 0.2 },
    ]);
  });
});

// The wizard feeds onChange back into `values`; mirror that so the controlled inputs behave as in the app.
function Harness({ onChange }: { onChange: (v: Record<string, unknown>) => void }) {
  const [values, setValues] = useState<Record<string, unknown>>({ ...defaultsOf(paramSpecs), vpr_match_seq_len: 12 });
  return (
    <ParamForm
      specs={paramSpecs}
      values={values}
      onChange={(v) => {
        setValues(v);
        onChange(v);
      }}
    />
  );
}

describe("ParamForm", () => {
  it("renders groups, marks modified fields and reports changes", async () => {
    const onChange = vi.fn();
    render(<Harness onChange={onChange} />);
    expect(screen.getByText("localization")).toBeInTheDocument();
    expect(screen.getByText("modified")).toBeInTheDocument();
    const sigma = screen.getByLabelText("pgo_loop_sigma_trans");
    await userEvent.clear(sigma);
    await userEvent.type(sigma, "0.2");
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ pgo_loop_sigma_trans: 0.2 }));
  });
});
