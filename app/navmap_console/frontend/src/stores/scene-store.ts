import { create } from "zustand";
import { subscribeWithSelector } from "zustand/middleware";

export type ColorMode = "focus" | "order" | "component" | "new";
export type CameraMode = "perspective" | "top";
export type UpAxis = "z" | "y";
export type NodeStyle = "auto" | "frustum" | "points";
export type LayerKey = "odom" | "covis" | "trav" | "loops" | "rejected" | "ghost" | "culled";
export type LoopFilter = "all" | "accepted" | "rejected" | "new" | "hist";
export type Selection = { kind: "node"; id: number } | { kind: "loop"; index: number } | null;
export type Vec3 = [number, number, number];

interface SceneData {
  step: number | null;
  maxStep: number;
  playing: boolean;
  speed: number;
  follow: boolean;
  colorMode: ColorMode;
  pinned: number[];
  layers: Record<LayerKey, boolean>;
  nodeStyle: NodeStyle;
  camera: CameraMode;
  up: UpAxis;
  morph: number;
  morphThreshold: number;
  selected: Selection;
  hovered: Selection;
  loopFilter: LoopFilter;
  flyTo: { target: Vec3; seq: number } | null;
  fitSeq: number;
}

interface SceneActions {
  setStep: (k: number) => void;
  stepBy: (delta: number) => void;
  setMaxStep: (m: number) => void;
  jumpToLatest: () => void;
  setPlaying: (on: boolean) => void;
  togglePlaying: () => void;
  setSpeed: (speed: number) => void;
  setFollow: (on: boolean) => void;
  setColorMode: (mode: ColorMode) => void;
  togglePin: (step: number) => void;
  setLayer: (key: LayerKey, on: boolean) => void;
  toggleLayer: (key: LayerKey) => void;
  setNodeStyle: (style: NodeStyle) => void;
  setCamera: (mode: CameraMode) => void;
  toggleCamera: () => void;
  setUp: (up: UpAxis) => void;
  setMorph: (t: number) => void;
  setMorphThreshold: (m: number) => void;
  select: (sel: Selection) => void;
  hover: (sel: Selection) => void;
  setLoopFilter: (f: LoopFilter) => void;
  requestFly: (target: Vec3) => void;
  requestFit: () => void;
  reset: () => void;
}

export type SceneState = SceneData & SceneActions;

const MAX_PINS = 5;
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

const initial: SceneData = {
  step: null, maxStep: -1, playing: false, speed: 1, follow: true, colorMode: "focus", pinned: [],
  layers: { odom: true, covis: false, trav: false, loops: true, rejected: true, ghost: false, culled: false },
  nodeStyle: "auto", camera: "perspective", up: "z", morph: 1, morphThreshold: 0.05,
  selected: null, hovered: null, loopFilter: "all", flyTo: null, fitSeq: 0,
};

export const useSceneStore = create<SceneState>()(
  subscribeWithSelector((set, get) => ({
    ...initial,
    setStep: (k) => set((s) => ({ step: clamp(k, 0, Math.max(0, s.maxStep)), follow: false, playing: false })),
    stepBy: (delta) => {
      const s = get();
      s.setStep((s.step ?? 0) + delta);
    },
    setMaxStep: (m) => set((s) => ({ maxStep: m, step: s.follow || s.step === null ? m : Math.min(s.step, m) })),
    jumpToLatest: () => set((s) => ({ step: s.maxStep, follow: true })),
    setPlaying: (playing) => set({ playing }),
    togglePlaying: () => set((s) => ({ playing: !s.playing })),
    setSpeed: (speed) => set({ speed }),
    setFollow: (follow) => set({ follow }),
    setColorMode: (colorMode) => set({ colorMode }),
    togglePin: (step) =>
      set((s) => {
        if (s.pinned.includes(step)) return { pinned: s.pinned.filter((p) => p !== step) };
        const next = [...s.pinned, step];
        return { pinned: next.slice(Math.max(0, next.length - MAX_PINS)) };
      }),
    setLayer: (key, on) => set((s) => ({ layers: { ...s.layers, [key]: on } })),
    toggleLayer: (key) => set((s) => ({ layers: { ...s.layers, [key]: !s.layers[key] } })),
    setNodeStyle: (nodeStyle) => set({ nodeStyle }),
    setCamera: (camera) => set({ camera }),
    toggleCamera: () => set((s) => ({ camera: s.camera === "top" ? "perspective" : "top" })),
    setUp: (up) => set({ up }),
    setMorph: (t) => set({ morph: clamp(t, 0, 1) }),
    setMorphThreshold: (morphThreshold) => set({ morphThreshold }),
    select: (selected) => set({ selected }),
    hover: (hovered) => set({ hovered }),
    setLoopFilter: (loopFilter) => set({ loopFilter }),
    requestFly: (target) => set((s) => ({ flyTo: { target, seq: (s.flyTo?.seq ?? 0) + 1 } })),
    requestFit: () => set((s) => ({ fitSeq: s.fitSeq + 1 })),
    reset: () => set({ ...initial, layers: { ...initial.layers } }),
  })),
);
