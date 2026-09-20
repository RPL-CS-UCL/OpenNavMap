import { useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import { EmptyState } from "@/components/common/EmptyState";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { t } from "@/i18n";

const PANELS = ["vpr", "loops", "pgo", "culling", "charts", "console"] as const;

/** Bottom dock skeleton; Tasks 12/13 fill the vpr/loops tabs and the remaining panels. */
export function PanelDock() {
  const [collapsed, setCollapsed] = useState(false);
  return (
    <div className="border-t bg-background">
      <div className="flex items-center gap-2 px-2">
        <Tabs defaultValue="vpr" className="w-full">
          <TabsList className="h-7">
            {PANELS.map((k) => (
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
          {!collapsed && PANELS.map((k) => (
            <TabsContent key={k} value={k} className="m-0 h-44">
              <EmptyState title={t(`viewer.dock.${k}`)} body={t("viewer.dock.placeholder")} />
            </TabsContent>
          ))}
        </Tabs>
      </div>
    </div>
  );
}
