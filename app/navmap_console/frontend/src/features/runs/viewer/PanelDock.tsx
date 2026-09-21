import { ChevronsUpDown } from "lucide-react";
import type { Scene } from "@/api/scene-bundle";
import type { StepSummary } from "@/api/types";
import { EmptyState } from "@/components/common/EmptyState";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { LogConsole } from "@/features/jobs/LogConsole";
import { t } from "@/i18n";
import { ChartsPanel } from "../panels/ChartsPanel";
import { CullingPanel } from "../panels/CullingPanel";
import { DMatrixPanel } from "../panels/DMatrixPanel";
import { LoopEdgesTable } from "../panels/LoopEdgesTable";
import { PgoSummaryPanel } from "../panels/PgoSummaryPanel";

/** Bottom dock: VPR matrix, loop edges, PGO summary, culling, charts and the job console. */
export function PanelDock({ rid, runId, step, scene, summaries, logLines, hasJob, onToggleCollapse }: {
  rid: string; runId: string; step: number; scene: Scene | null;
  summaries: StepSummary[] | undefined; logLines: string[]; hasJob: boolean;
  /** Collapses / expands the resizable panel hosting the dock (owned by RunViewer). */
  onToggleCollapse?: () => void;
}) {
  return (
    <div className="flex h-full min-h-0 flex-col border-t bg-background">
      <div className="flex min-h-0 flex-1 gap-2 px-2">
        <Tabs defaultValue="vpr" className="flex min-h-0 w-full flex-col">
          <TabsList className="h-7">
            <TabsTrigger value="vpr">{t("viewer.dock.vpr")}</TabsTrigger>
            <TabsTrigger value="loops">{t("viewer.dock.loops")}</TabsTrigger>
            <TabsTrigger value="pgo">{t("viewer.dock.pgo")}</TabsTrigger>
            <TabsTrigger value="culling">{t("viewer.dock.culling")}</TabsTrigger>
            <TabsTrigger value="charts">{t("viewer.dock.charts")}</TabsTrigger>
            <TabsTrigger value="console">{t("viewer.dock.console")}</TabsTrigger>
          </TabsList>
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6"
            aria-label={t("viewer.dock.collapse")}
            onClick={onToggleCollapse}
          >
            <ChevronsUpDown />
          </Button>
          <TabsContent value="vpr" className="m-0 min-h-0 flex-1">
            <DMatrixPanel rid={rid} runId={runId} step={step} scene={scene} />
          </TabsContent>
          <TabsContent value="loops" className="m-0 min-h-0 flex-1">
            <LoopEdgesTable rid={rid} runId={runId} scene={scene} />
          </TabsContent>
          <TabsContent value="pgo" className="m-0 min-h-0 flex-1">
            <PgoSummaryPanel summary={summaries?.find((s) => s.index === step)} scene={scene} />
          </TabsContent>
          <TabsContent value="culling" className="m-0 min-h-0 flex-1">
            <CullingPanel rid={rid} runId={runId} step={step} />
          </TabsContent>
          <TabsContent value="charts" className="m-0 min-h-0 flex-1">
            <ChartsPanel summaries={summaries ?? []} step={step} />
          </TabsContent>
          <TabsContent value="console" className="m-0 min-h-0 flex-1">
            {hasJob ? (
              // jsdom has no layout; Virtuoso needs an item count hint to render rows in tests.
              <LogConsole lines={logLines} initialItemCount={Math.max(logLines.length, 1)} />
            ) : (
              <EmptyState title={t("panel.console.none")} body={t("panel.console.imported")} />
            )}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
