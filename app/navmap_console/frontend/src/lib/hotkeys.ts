import { useEffect } from "react";

interface Opts {
  ctrl?: boolean;
  enabled?: boolean;
}

/** Registers a keydown handler on window; ignored while typing in inputs unless ctrl is required. */
export function useHotkey(key: string, handler: () => void, opts: Opts = {}): void {
  const { ctrl = false, enabled = true } = opts;
  useEffect(() => {
    if (!enabled) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() !== key.toLowerCase()) return;
      if (ctrl !== (e.ctrlKey || e.metaKey)) return;
      const el = e.target as HTMLElement | null;
      const typing = !!el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable);
      if (typing && !ctrl) return;
      e.preventDefault();
      handler();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [key, ctrl, enabled, handler]);
}
