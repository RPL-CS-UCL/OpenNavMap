import { useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import type { Scene } from "@/api/scene-bundle";
import { EmptyState } from "@/components/common/EmptyState";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { t } from "@/i18n";
import { DMatrixPanel } from "../panels/DMatrixPanel";
import { LoopEdgesTable } from "../panels/LoopEdgesTable";

const PLACEHOLDERS = ["pgo", "culling", "charts", "console"] as const;

/** Bottom dock: VPR matrix and loop edge table arrive here; the rest is still placeholder. */
export function PanelDock({ rid, runId, step, scene }: {
  rid: string; runId: string; step: number; scene: Scene | null;
}) {
  const [collapsed, setCollapsed] = useState(false);
  return (
    <div className="border-t bg-background">
      <div className="flex items-center gap-2 px-2">
        <Tabs defaultValue="vpr" className="w-full">
          <TabsList className="h-7">
            <TabsTrigger value="vpr">{t("viewer.dock.vpr")}</TabsTrigger>
            <TabsTrigger value="loops">{t("viewer.dock.loops")}</TabsTrigger>
            {PLACEHOLDERS.map((k) => (
              <TabsTrigger key={k} value={k}>{t(`viewer.dock.${k}`)}</TabsTrigger>
            ))}
          </TabsList>
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6"
            aria-label={t("viewer.dock.collapse")}
            onClick={() => setCollapsed((c) => !c)}
          >
            {collapsed ? <ChevronUp /> : <ChevronDown />}
          </Button>
          {!collapsed && (
            <>
              <TabsContent value="vpr" className="m-0 h-44">
                <DMatrixPanel rid={rid} runId={runId} step={step} scene={scene} />
              </TabsContent>
              <TabsContent value="loops" className="m-0 h-44">
                <LoopEdgesTable rid={rid} runId={runId} scene={scene} />
              </TabsContent>
              {PLACEHOLDERS.map((k) => (
                <TabsContent key={k} value={k} className="m-0 h-44">
                  <EmptyState title={t(`viewer.dock.${k}`)} body={t("viewer.dock.placeholder")} />
                </TabsContent>
              ))}
            </>
          )}
        </Tabs>
      </div>
    </div>
  );
}
