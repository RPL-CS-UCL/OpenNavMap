import { useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";
import { prefetchScene, useScene, useStepSummaries } from "@/api/hooks/use-results";
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";
import { SceneCanvas } from "@/scene/SceneCanvas";
import { SceneLegend } from "@/scene/SceneLegend";
import { useSceneStore } from "@/stores/scene-store";
import { Inspector } from "./Inspector";
import { PanelDock } from "./PanelDock";
import { SceneToolbar } from "./SceneToolbar";
import { StepList } from "./StepList";
import { StepSlider } from "./StepSlider";
import { useStepUrl } from "./use-step-url";
import { useViewerHotkeys } from "./use-viewer-hotkeys";

export function RunViewer({ rid, runId }: { rid: string; runId: string }) {
  const qc = useQueryClient();
  const step = useSceneStore((s) => s.step);
  const summaries = useStepSummaries(rid, runId);
  const scene = useScene(rid, runId, step);

  // A different run: fresh viewer state before the URL effect re-applies any deep link.
  useEffect(() => {
    useSceneStore.getState().reset();
  }, [rid, runId]);
  useStepUrl();
  useViewerHotkeys();

  // Summaries define the step axis; follow mode (or no step yet) lands on the newest step.
  useEffect(() => {
    if (summaries.data) useSceneStore.getState().setMaxStep(summaries.data.length - 1);
  }, [summaries.data]);

  // Prefetch neighbours so scrubbing the slider feels instant.
  useEffect(() => {
    if (step === null) return;
    void prefetchScene(qc, rid, runId, step - 1);
    void prefetchScene(qc, rid, runId, step + 1);
  }, [qc, rid, runId, step]);

  const hasPrePgo = summaries.data?.[step ?? -1]?.has_pre_pgo ?? false;

  return (
    <div className="flex h-full min-h-0 flex-col">
      <ResizablePanelGroup orientation="horizontal" className="min-h-0 flex-1">
        <ResizablePanel defaultSize={20} minSize={14}>
          <StepList steps={summaries.data} pending={summaries.isPending} />
        </ResizablePanel>
        <ResizableHandle />
        <ResizablePanel defaultSize={58} minSize={30} className="flex min-h-0 flex-col">
          <SceneToolbar />
          <div className="relative min-h-0 flex-1">
            <SceneCanvas scene={scene.data ?? null} />
            <div className="absolute bottom-2 left-2">
              <SceneLegend scene={scene.data ?? null} />
            </div>
          </div>
          <StepSlider hasPrePgo={hasPrePgo} />
        </ResizablePanel>
        <ResizableHandle />
        <ResizablePanel defaultSize={22} minSize={16}>
          <Inspector rid={rid} runId={runId} scene={scene.data ?? null} />
        </ResizablePanel>
      </ResizablePanelGroup>
      <PanelDock rid={rid} runId={runId} step={step ?? 0} scene={scene.data ?? null} />
    </div>
  );
}
