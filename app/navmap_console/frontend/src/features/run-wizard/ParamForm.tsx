import type { ReactElement } from "react";
import { useMemo, useState } from "react";
import type { ParamSpec } from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { t } from "@/i18n";
import { GROUP_ORDER } from "./param-schema";

interface Props {
  specs: ParamSpec[];
  values: Record<string, unknown>;
  onChange: (values: Record<string, unknown>) => void;
}

function formatNumber(value: unknown): string {
  return value === undefined || value === null ? "" : String(value);
}

function parseNumber(spec: ParamSpec, raw: string): unknown {
  if (raw === "") return undefined;
  const n = spec.type === "int" ? parseInt(raw, 10) : parseFloat(raw);
  // Keep the raw text when unparsable so zod flags it instead of silently dropping the field.
  return Number.isNaN(n) ? raw : n;
}

function NumberField({ id, spec, value, onChange }: { id: string; spec: ParamSpec; value: unknown; onChange: (v: unknown) => void }) {
  // Keep the raw text locally so a partial decimal like "0." survives the round trip through the parent,
  // which only ever sees parsed numbers; resync when the parent hands back a different value (e.g. reset).
  const [local, setLocal] = useState({ text: formatNumber(value), value });
  if (local.value !== value) setLocal({ text: formatNumber(value), value });
  return (
    <Input
      id={id}
      type="number"
      step={spec.type === "int" ? 1 : "any"}
      className="h-8 w-56 font-mono"
      value={local.text}
      onChange={(e) => {
        const parsed = parseNumber(spec, e.target.value);
        setLocal({ text: e.target.value, value: parsed });
        onChange(parsed);
      }}
      aria-label={spec.name}
    />
  );
}

function Field({ spec, value, onChange }: { spec: ParamSpec; value: unknown; onChange: (v: unknown) => void }) {
  const id = `param-${spec.name}`;
  const modified = value !== spec.default;
  let control: ReactElement;
  if (spec.type === "bool") {
    control = <Switch id={id} checked={Boolean(value)} onCheckedChange={(v) => onChange(v)} aria-label={spec.name} />;
  } else if (spec.type === "choice") {
    control = (
      <Select value={String(value)} onValueChange={(v) => onChange(v)}>
        <SelectTrigger id={id} className="h-8 w-56" aria-label={spec.name}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {(spec.choices ?? []).map((c) => (
            <SelectItem key={c} value={c}>
              {c}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    );
  } else if (spec.type === "int" || spec.type === "float") {
    control = <NumberField id={id} spec={spec} value={value} onChange={onChange} />;
  } else {
    control = <Input id={id} className="h-8 w-56 font-mono" value={String(value ?? "")} onChange={(e) => onChange(e.target.value)} aria-label={spec.name} />;
  }
  return (
    <div className="grid grid-cols-[220px_auto_1fr] items-center gap-3 py-1.5">
      <Label htmlFor={id} className="font-mono text-xs">
        {spec.name}
        {modified && (
          <Badge variant="outline" className="ml-2 h-4 px-1 text-[10px]">
            {t("wizard.params.modified")}
          </Badge>
        )}
      </Label>
      {control}
      <div className="text-xs text-muted-foreground">
        {spec.help}
        <span className="ml-2 font-mono text-[11px]">{t("wizard.params.default", { value: String(spec.default) })}</span>
      </div>
    </div>
  );
}

export function ParamForm({ specs, values, onChange }: Props) {
  const groups = useMemo(() => {
    const by = new Map<string, ParamSpec[]>();
    for (const s of specs) by.set(s.group, [...(by.get(s.group) ?? []), s]);
    return [...by.entries()].sort(([a], [b]) => GROUP_ORDER.indexOf(a) - GROUP_ORDER.indexOf(b));
  }, [specs]);
  const set = (name: string, v: unknown) => onChange({ ...values, [name]: v });

  return (
    <div className="space-y-4">
      {groups.map(([group, list]) => {
        const advanced = list.every((s) => s.advanced);
        const body = list.map((s) => <Field key={s.name} spec={s} value={values[s.name]} onChange={(v) => set(s.name, v)} />);
        return advanced ? (
          <Collapsible key={group}>
            <CollapsibleTrigger className="text-sm font-semibold">{group} (advanced)</CollapsibleTrigger>
            <CollapsibleContent>{body}</CollapsibleContent>
          </Collapsible>
        ) : (
          <section key={group}>
            <h3 className="mb-1 text-sm font-semibold">{group}</h3>
            {body}
          </section>
        );
      })}
    </div>
  );
}
