import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { toast } from "sonner";
import { errorDetail } from "@/api/client";
import { useImportRun } from "@/api/hooks/use-results";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { FsBrowserDialog } from "@/features/sessions/FsBrowserDialog";
import { t } from "@/i18n";

export function ImportRunDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (o: boolean) => void }) {
  const { rid = "" } = useParams();
  const navigate = useNavigate();
  const importRun = useImportRun(rid);
  const [name, setName] = useState("");
  const [resultDir, setResultDir] = useState("");
  const [sessionsRoot, setSessionsRoot] = useState("");
  const [promote, setPromote] = useState(false);
  const [browseTarget, setBrowseTarget] = useState<"result" | "sessions" | null>(null);

  const submit = () => {
    importRun.mutate(
      { result_dir: resultDir, sessions_root: sessionsRoot || undefined, name: name || undefined, promote },
      {
        onSuccess: (run) => {
          toast.success(t("run.import.imported", { name: run.name || run.id }));
          onOpenChange(false);
          navigate(`/regions/${rid}/runs/${run.id}`);
        },
        onError: (err) => toast.error(errorDetail(err)),
      },
    );
  };

  return (
    <>
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t("run.import.title")}</DialogTitle>
          </DialogHeader>
          <div className="grid gap-3">
            <div className="grid gap-1.5">
              <Label htmlFor="import-name">{t("run.import.name")}</Label>
              <Input id="import-name" value={name} onChange={(e) => setName(e.target.value)} />
            </div>
            <div className="grid gap-1.5">
              <Label htmlFor="import-result">{t("run.import.resultDir")}</Label>
              <div className="flex gap-1">
                <Input id="import-result" value={resultDir} onChange={(e) => setResultDir(e.target.value)} placeholder="/Titan/dataset/…" />
                <Button type="button" size="sm" variant="outline" onClick={() => setBrowseTarget("result")}>
                  {t("common.browse")}
                </Button>
              </div>
            </div>
            <div className="grid gap-1.5">
              <Label htmlFor="import-sessions">{t("run.import.sessionsRoot")}</Label>
              <div className="flex gap-1">
                <Input id="import-sessions" value={sessionsRoot} onChange={(e) => setSessionsRoot(e.target.value)} />
                <Button type="button" size="sm" variant="outline" onClick={() => setBrowseTarget("sessions")}>
                  {t("common.browse")}
                </Button>
              </div>
              <p className="text-xs text-muted-foreground">{t("run.import.sessionsHint")}</p>
            </div>
            <div className="flex items-center gap-2">
              <Switch id="import-promote" checked={promote} onCheckedChange={setPromote} />
              <Label htmlFor="import-promote">{t("run.import.promote")}</Label>
            </div>
          </div>
          <DialogFooter>
            <Button size="sm" variant="outline" onClick={() => onOpenChange(false)}>
              {t("common.cancel")}
            </Button>
            <Button size="sm" disabled={!resultDir || importRun.isPending} onClick={submit}>
              {t("run.import.button")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      <FsBrowserDialog
        open={browseTarget !== null}
        onOpenChange={(o) => !o && setBrowseTarget(null)}
        onChoose={(path) => {
          if (browseTarget === "result") setResultDir(path);
          else if (browseTarget === "sessions") setSessionsRoot(path);
          setBrowseTarget(null);
        }}
      />
    </>
  );
}
