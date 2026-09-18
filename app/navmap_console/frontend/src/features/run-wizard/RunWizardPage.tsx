import { useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { toast } from "sonner";
import { errorDetail } from "@/api/client";
import { useMergeParams } from "@/api/hooks/use-params";
import { useRegion } from "@/api/hooks/use-regions";
import { useCreateRun, useRuns } from "@/api/hooks/use-runs";
import { useSessions } from "@/api/hooks/use-sessions";
import { PageHeader } from "@/components/common/PageHeader";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { t } from "@/i18n";
import { cn } from "@/lib/utils";
import { ModeStep } from "./ModeStep";
import { ParamsStep } from "./ParamsStep";
import { ReviewStep } from "./ReviewStep";
import { SessionOrderStep } from "./SessionOrderStep";
import { buildParamSchema, defaultsOf } from "./param-schema";
import { type WizardStep, useWizardStore } from "./wizard-store";

const STEP_KEYS = ["wizard.step.sessions", "wizard.step.mode", "wizard.step.params", "wizard.step.review"] as const;

export function RunWizardPage() {
  const { rid = "" } = useParams();
  const navigate = useNavigate();
  const w = useWizardStore();
  const region = useRegion(rid);
  const sessions = useSessions(rid);
  const runs = useRuns(rid);
  const specs = useMergeParams();
  const create = useCreateRun(rid);
  const setRid = w.setRid;
  const setParams = w.setParams;
  const paramsEmpty = Object.keys(w.params).length === 0;

  useEffect(() => setRid(rid), [rid, setRid]);
  useEffect(() => {
    if (specs.data && paramsEmpty) setParams(defaultsOf(specs.data));
  }, [specs.data, paramsEmpty, setParams]);

  if (region.isPending || sessions.isPending || specs.isPending || runs.isPending) return <Skeleton className="h-40 w-full" />;
  if (region.error || sessions.error || specs.error || runs.error)
    return <p className="text-sm text-destructive">{t("common.error", { detail: "load failed" })}</p>;

  const paramsOk = buildParamSchema(specs.data).safeParse(w.params).success;
  const canNext = [w.order.length > 0, true, paramsOk, true][w.step];

  const start = () =>
    create.mutate(
      {
        name: w.name || undefined,
        kind: w.mode,
        parent: w.mode === "append" ? w.parent : null,
        session_ids: w.order,
        params: w.params,
        meta: w.seed !== null ? { shuffle_seed: w.seed } : {},
      },
      {
        onSuccess: (run) => {
          toast.success(t("wizard.created", { name: run.name || run.id }));
          w.reset();
          navigate(`/regions/${rid}/runs/${run.id}`);
        },
        onError: (err) => toast.error(errorDetail(err)),
      },
    );

  return (
    <div>
      <PageHeader backTo={`/regions/${rid}`} title={t("wizard.title")} description={region.data.name} />
      <ol className="mb-4 flex gap-4 text-xs">
        {STEP_KEYS.map((k, i) => (
          <li key={k} className={cn("flex items-center gap-1", i === w.step ? "font-semibold text-foreground" : "text-muted-foreground")}>
            <span className="flex h-4 w-4 items-center justify-center rounded-full border font-mono text-[10px]">{i + 1}</span>
            {t(k)}
          </li>
        ))}
      </ol>
      {w.step === 0 && <SessionOrderStep sessions={sessions.data} order={w.order} seed={w.seed} onOrder={w.setOrder} onSeed={w.setSeed} />}
      {w.step === 1 && <ModeStep region={region.data} runs={runs.data} mode={w.mode} parent={w.parent} onMode={w.setMode} onParent={w.setParent} />}
      {w.step === 2 && <ParamsStep specs={specs.data} values={w.params} onChange={w.setParams} />}
      {w.step === 3 && (
        <ReviewStep
          specs={specs.data}
          sessions={sessions.data}
          order={w.order}
          mode={w.mode}
          parent={w.parent}
          params={w.params}
          name={w.name}
          runs={runs.data}
          onName={w.setName}
        />
      )}
      <div className="mt-6 flex gap-2">
        <Button variant="outline" size="sm" disabled={w.step === 0} onClick={() => w.setStep((w.step - 1) as WizardStep)}>
          {t("wizard.back")}
        </Button>
        {w.step < 3 ? (
          <Button size="sm" disabled={!canNext} onClick={() => w.setStep((w.step + 1) as WizardStep)}>
            {t("wizard.next")}
          </Button>
        ) : (
          <Button size="sm" disabled={create.isPending} onClick={start}>
            {t("wizard.start")}
          </Button>
        )}
      </div>
    </div>
  );
}
