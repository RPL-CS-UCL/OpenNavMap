import { create } from "zustand";

interface JobDockState {
  open: boolean;
  activeJobId: string | null;
  setOpen: (open: boolean) => void;
  toggle: () => void;
  setActive: (jobId: string | null) => void;
}

export const useJobDockStore = create<JobDockState>((set) => ({
  open: false,
  activeJobId: null,
  setOpen: (open) => set({ open }),
  toggle: () => set((s) => ({ open: !s.open })),
  setActive: (activeJobId) => set({ activeJobId, open: true }),
}));
