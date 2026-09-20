import { LOOP_ACCEPTED, NODE_CULLED, NODE_NEW } from "@/api/scene-bundle";
import type { ColorMode } from "@/stores/scene-store";

export type RGB = [number, number, number];

export interface Palette {
  ref: RGB; current: RGB; accept: RGB; reject: RGB; ghost: RGB; culled: RGB; select: RGB;
  pins: RGB[]; seq: RGB[]; series: RGB[];
}

/** "215 80% 48%" (shadcn token format) -> sRGB in 0..1. */
export function parseHslTriple(s: string): RGB {
  const [h, sp, lp] = s.trim().split(/\s+/);
  const hue = (parseFloat(h) % 360) / 360, sat = parseFloat(sp) / 100, lig = parseFloat(lp) / 100;
  if (sat === 0) return [lig, lig, lig];
  const q = lig < 0.5 ? lig * (1 + sat) : lig + sat - lig * sat;
  const p = 2 * lig - q;
  const f = (t: number) => {
    if (t < 0) t += 1;
    if (t > 1) t -= 1;
    if (t < 1 / 6) return p + (q - p) * 6 * t;
    if (t < 1 / 2) return q;
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
    return p;
  };
  return [f(hue + 1 / 3), f(hue), f(hue - 1 / 3)];
}

export function srgbToLinear(c: RGB): RGB {
  return c.map((v) => (v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4)) as RGB;
}

export const FALLBACK_TOKENS: Record<string, string> = {
  "scene-ref": "0 0% 58%", "scene-current": "28 92% 50%", "scene-accept": "145 60% 36%", "scene-reject": "0 75% 48%",
  "scene-ghost": "0 0% 72%", "scene-culled": "0 0% 35%", "scene-select": "48 100% 45%",
  "scene-pin-1": "215 80% 48%", "scene-pin-2": "270 55% 55%", "scene-pin-3": "180 55% 38%", "scene-pin-4": "330 60% 52%",
  "scene-pin-5": "240 50% 55%", "scene-bg": "40 20% 94%",
  "seq-100": "212 90% 92%", "seq-200": "212 85% 82%", "seq-300": "212 80% 70%", "seq-400": "212 75% 58%",
  "seq-500": "212 75% 46%", "seq-600": "212 75% 36%", "seq-700": "212 75% 26%",
  "series-1": "215 80% 48%", "series-2": "28 90% 50%", "series-3": "180 55% 38%",
};

export function readToken(name: string, el: Element = document.documentElement): RGB {
  const raw = getComputedStyle(el).getPropertyValue(`--${name}`).trim() || FALLBACK_TOKENS[name] || "0 0% 50%";
  return srgbToLinear(parseHslTriple(raw));
}

export function readPalette(el: Element = document.documentElement): Palette {
  const t = (n: string) => readToken(n, el);
  return {
    ref: t("scene-ref"), current: t("scene-current"), accept: t("scene-accept"), reject: t("scene-reject"),
    ghost: t("scene-ghost"), culled: t("scene-culled"), select: t("scene-select"),
    pins: [1, 2, 3, 4, 5].map((i) => t(`scene-pin-${i}`)),
    seq: [100, 200, 300, 400, 500, 600, 700].map((i) => t(`seq-${i}`)),
    series: [1, 2, 3].map((i) => t(`series-${i}`)),
  };
}

export function seqColor(t: number, seq: RGB[]): RGB {
  const x = Math.min(1, Math.max(0, t)) * (seq.length - 1);
  const i = Math.min(seq.length - 2, Math.floor(x));
  const f = x - i;
  return [0, 1, 2].map((k) => seq[i][k] + (seq[i + 1][k] - seq[i][k]) * f) as RGB;
}

interface NodeArrays { numNodes: number; step: Uint16Array; comp: Uint16Array; flags: Uint8Array }

export function nodeColors(scene: NodeArrays, mode: ColorMode, currentStep: number, pinned: number[], palette: Palette, out?: Float32Array): Float32Array {
  const n = scene.numNodes;
  const c = out && out.length >= n * 3 ? out : new Float32Array(n * 3);
  const denom = Math.max(1, currentStep);
  for (let i = 0; i < n; i++) {
    let rgb: RGB;
    const step = scene.step[i];
    if (scene.flags[i] & NODE_CULLED) rgb = palette.culled;
    else if (mode === "order") rgb = seqColor(step / denom, palette.seq);
    else if (mode === "component") rgb = scene.comp[i] < palette.series.length ? palette.series[scene.comp[i]] : palette.ghost;
    else if (mode === "new") rgb = scene.flags[i] & NODE_NEW ? palette.current : palette.ref;
    else {
      const pin = pinned.indexOf(step);
      rgb = step === currentStep ? palette.current : pin >= 0 ? palette.pins[pin % palette.pins.length] : palette.ref;
    }
    c[i * 3] = rgb[0]; c[i * 3 + 1] = rgb[1]; c[i * 3 + 2] = rgb[2];
  }
  return c;
}

/** Accepted = accept colour, everything else = reject; low GNC weight fades toward ghost. */
export function loopColor(flags: number, weight: number, palette: Palette): RGB {
  const base = flags & LOOP_ACCEPTED ? palette.accept : palette.reject;
  const w = 0.35 + 0.65 * Math.min(1, Math.max(0, weight));
  return [0, 1, 2].map((k) => palette.ghost[k] + (base[k] - palette.ghost[k]) * w) as RGB;
}
