import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import type { RunParent } from "@/api/types";

export type WizardStep = 0 | 1 | 2 | 3;
type Mode = "merge" | "append";

interface WizardState {
  rid: string;
  step: WizardStep;
  order: string[];
  mode: Mode;
  parent: RunParent | null;
  params: Record<string, unknown>;
  name: string;
  seed: number | null;
  setRid: (rid: string) => void;
  setStep: (step: WizardStep) => void;
  setOrder: (order: string[]) => void;
  setMode: (mode: Mode) => void;
  setParent: (parent: RunParent | null) => void;
  setParams: (params: Record<string, unknown>) => void;
  setName: (name: string) => void;
  setSeed: (seed: number | null) => void;
  reset: () => void;
}

const initial = {
  rid: "",
  step: 0 as WizardStep,
  order: [] as string[],
  mode: "merge" as Mode,
  parent: null as RunParent | null,
  params: {} as Record<string, unknown>,
  name: "",
  seed: null as number | null,
};

// Persisted per tab so a page refresh keeps the picked order; switching region starts over.
export const useWizardStore = create<WizardState>()(
  persist(
    (set) => ({
      ...initial,
      setRid: (rid) => set((s) => (s.rid === rid ? {} : { ...initial, rid })),
      setStep: (step) => set({ step }),
      setOrder: (order) => set({ order }),
      setMode: (mode) => set({ mode }),
      setParent: (parent) => set({ parent }),
      setParams: (params) => set({ params }),
      setName: (name) => set({ name }),
      setSeed: (seed) => set({ seed }),
      reset: () => set({ ...initial }),
    }),
    { name: "navmap.run-wizard", storage: createJSONStorage(() => sessionStorage) },
  ),
);
