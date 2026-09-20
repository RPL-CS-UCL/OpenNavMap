import { useEffect, useState } from "react";
import { FALLBACK_TOKENS, type Palette, readPalette } from "./build/colors";

export interface ScenePalette { palette: Palette; background: string }

function read(): ScenePalette {
  const root = document.documentElement;
  const bg = getComputedStyle(root).getPropertyValue("--scene-bg").trim() || FALLBACK_TOKENS["scene-bg"];
  return { palette: readPalette(root), background: `hsl(${bg})` };
}

/** Palette from CSS tokens; re-read when the theme class on <html> changes. */
export function useScenePalette(): ScenePalette {
  const [state, setState] = useState<ScenePalette>(read);
  useEffect(() => {
    const obs = new MutationObserver(() => setState(read()));
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["class", "style"] });
    return () => obs.disconnect();
  }, []);
  return state;
}
