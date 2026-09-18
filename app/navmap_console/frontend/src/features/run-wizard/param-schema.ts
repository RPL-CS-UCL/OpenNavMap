import { z } from "zod";
import type { ParamSpec } from "@/api/types";

/** Display order of parameter groups; unknown groups sort last. */
export const GROUP_ORDER = ["localization", "optimization", "culling", "advanced"];

export function buildParamSchema(specs: ParamSpec[]) {
  const shape: Record<string, z.ZodTypeAny> = {};
  for (const s of specs) {
    switch (s.type) {
      case "int":
        shape[s.name] = z.number().int();
        break;
      case "float":
        shape[s.name] = z.number();
        break;
      case "bool":
        shape[s.name] = z.boolean();
        break;
      case "choice":
        shape[s.name] = z.enum((s.choices ?? []) as [string, ...string[]]);
        break;
      default:
        shape[s.name] = z.string();
    }
  }
  return z.object(shape);
}

export function defaultsOf(specs: ParamSpec[]): Record<string, unknown> {
  return Object.fromEntries(specs.map((s) => [s.name, s.default]));
}

export function diffParams(specs: ParamSpec[], values: Record<string, unknown>): Array<{ name: string; from: unknown; to: unknown }> {
  return specs
    .filter((s) => values[s.name] !== undefined && values[s.name] !== s.default)
    .map((s) => ({ name: s.name, from: s.default, to: values[s.name] }));
}
