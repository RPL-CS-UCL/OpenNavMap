import type { ParamSpec } from "@/api/types";
import { Button } from "@/components/ui/button";
import { t } from "@/i18n";
import { ParamForm } from "./ParamForm";
import { defaultsOf } from "./param-schema";

interface Props {
  specs: ParamSpec[];
  values: Record<string, unknown>;
  onChange: (v: Record<string, unknown>) => void;
}

export function ParamsStep({ specs, values, onChange }: Props) {
  return (
    <div className="space-y-3">
      <div className="flex justify-end">
        <Button size="sm" variant="ghost" onClick={() => onChange(defaultsOf(specs))}>
          {t("wizard.params.reset")}
        </Button>
      </div>
      <ParamForm specs={specs} values={values} onChange={onChange} />
    </div>
  );
}
