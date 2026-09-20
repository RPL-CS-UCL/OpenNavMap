import { useHotkey } from "@/lib/hotkeys";
import { useSceneStore } from "@/stores/scene-store";

/** Viewer-wide keyboard shortcuts; handlers read the store lazily so they stay referentially stable. */
export function useViewerHotkeys(): void {
  useHotkey("[", () => useSceneStore.getState().stepBy(-1));
  useHotkey("]", () => useSceneStore.getState().stepBy(1));
  useHotkey(" ", () => useSceneStore.getState().togglePlaying());
  useHotkey("f", () => useSceneStore.getState().requestFit());
  useHotkey("t", () => useSceneStore.getState().toggleCamera());
  useHotkey("1", () => useSceneStore.getState().toggleLayer("odom"));
  useHotkey("2", () => useSceneStore.getState().toggleLayer("covis"));
  useHotkey("3", () => useSceneStore.getState().toggleLayer("trav"));
  useHotkey("4", () => useSceneStore.getState().toggleLayer("loops"));
  useHotkey("g", () => useSceneStore.getState().toggleLayer("ghost"));
  useHotkey("Escape", () => useSceneStore.getState().select(null));
}
