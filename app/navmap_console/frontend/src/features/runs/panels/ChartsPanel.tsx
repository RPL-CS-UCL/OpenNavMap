import { useState, type ReactNode } from "react";
import { Bar, ComposedChart, Line, LineChart, ReferenceLine, XAxis, type MouseHandlerDataParam } from "recharts";
import type { StepSummary } from "@/api/types";
import { Toggle } from "@/components/ui/toggle";
import { t } from "@/i18n";
import { useSceneStore } from "@/stores/scene-store";

type Row = Record<string, number | null>;

/** PGO error per step; null means the step has no value (step 0, or the step is still running). */
export function pgoErrorRows(summaries: StepSummary[]): Row[] {
  return summaries.map((s) => ({ step: s.index, initial: s.pgo_error_initial, final: s.pgo_error_final }));
}

export function nodeComponentRows(summaries: StepSummary[]): Row[] {
  return summaries.map((s) => ({ step: s.index, num_nodes: s.num_nodes, num_components: s.component_sizes.length }));
}

/** Edge funnel: how candidate pairs (vpr) survive gv, ccm and pgo, and how many are retained. */
export function edgeSurvivalRows(summaries: StepSummary[]): Row[] {
  return summaries.map((s) => {
    const h = s.history;
    const vpr = h.vpr ?? 0, gv = h.gv ?? 0, ccm = h.ccm ?? 0, pgo = h.pgo ?? 0;
    return {
      step: s.index,
      retained: h.retained ?? 0,
      removedByGv: vpr - gv,
      removedByCcm: gv - ccm,
      removedByPgo: ccm - pgo,
    };
  });
}

export function durationRows(summaries: StepSummary[]): Row[] {
  return summaries.map((s) => ({ step: s.index, duration_s: s.duration_s }));
}

export function newCulledRows(summaries: StepSummary[]): Row[] {
  return summaries.map((s) => ({ step: s.index, num_new: s.num_new, num_culled: s.num_culled }));
}

/** Per-step ATE from the traj_evaluation toolchain; null until that step's eval job lands. */
export function ateRows(summaries: StepSummary[]): Row[] {
  return summaries.map((s) => ({ step: s.index, ate_trans_rmse: s.ate_trans_rmse, ate_rot_rmse: s.ate_rot_rmse }));
}

const SERIES = (n: number) => `hsl(var(--series-${n}))`;

/** X-axis tick that jumps the viewer to that step when clicked. */
function JumpTick({ x, y, payload, onStep }: { x?: number; y?: number; payload?: { value?: number }; onStep: (k: number) => void }) {
  const value = payload?.value;
  if (value === undefined) return null;
  return (
    <text
      x={x}
      y={y}
      dy={10}
      textAnchor="middle"
      fontSize={9}
      fill="hsl(var(--muted-foreground))"
      data-testid={`chart-tick-${value}`}
      onClick={() => onStep(value)}
      style={{ cursor: "pointer" }}
    >
      {value}
    </text>
  );
}

function Cell({ title, rows, cols, children }: {
  title: string; rows: Row[]; cols: string[]; children: ReactNode;
}) {
  const [asTable, setAsTable] = useState(false);
  const shownCols = cols.slice(0, 5);
  return (
    <div className="flex min-w-0 flex-col gap-0.5 rounded-sm border p-1">
      <div className="flex items-center justify-between gap-1">
        <span className="truncate text-muted-foreground">{title}</span>
        <Toggle size="sm" className="h-5 px-1 text-[10px]" pressed={asTable} onPressedChange={setAsTable}>
          {asTable ? t("panel.charts.chart") : t("panel.charts.table")}
        </Toggle>
      </div>
      {asTable ? (
        <table className="w-full border-collapse text-[10px]">
          <thead>
            <tr>{shownCols.map((c) => <th key={c} className="border px-1 text-left font-medium">{c}</th>)}</tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.step}>
                {shownCols.map((c) => <td key={c} className="border px-1 font-mono">{r[c] === null || r[c] === undefined ? "–" : r[c]}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        children
      )}
    </div>
  );
}

/** 3x2 grid of small per-step charts (recharts, fixed size). */
export function ChartsPanel({ summaries, step: _step }: { summaries: StepSummary[]; step: number }) {
  const [activeStep, setActiveStep] = useState<number | null>(null);
  const onStep = (k: number) => useSceneStore.getState().setStep(k);
  const tick = <JumpTick onStep={onStep} />;
  const onMove = (nextState: MouseHandlerDataParam) => {
    if (typeof nextState.activeTooltipIndex === "number") setActiveStep(nextState.activeTooltipIndex);
  };
  const activeRef = activeStep === null ? null : (
    <ReferenceLine x={activeStep} stroke="hsl(var(--border))" strokeDasharray="3 2" />
  );
  return (
    <div className="grid h-full min-h-0 grid-cols-3 gap-1 overflow-auto p-1 text-xs">
      <Cell title={t("panel.charts.pgoError")} rows={pgoErrorRows(summaries)} cols={["step", "initial", "final"]}>
        <LineChart width={170} height={88} data={pgoErrorRows(summaries)} onMouseMove={onMove}>
          <XAxis dataKey="step" interval={0} tick={tick} tickLine={false} axisLine={false} />
          {activeRef}
          <Line dataKey="initial" stroke={SERIES(1)} dot={false} isAnimationActive={false} connectNulls />
          <Line dataKey="final" stroke={SERIES(2)} dot={false} isAnimationActive={false} connectNulls />
        </LineChart>
      </Cell>
      <Cell title={t("panel.charts.nodes")} rows={nodeComponentRows(summaries)} cols={["step", "num_nodes", "num_components"]}>
        <ComposedChart width={170} height={88} data={nodeComponentRows(summaries)} onMouseMove={onMove}>
          <XAxis dataKey="step" interval={0} tick={tick} tickLine={false} axisLine={false} />
          {activeRef}
          <Line dataKey="num_nodes" stroke={SERIES(3)} dot={false} isAnimationActive={false} />
          <Bar dataKey="num_components" fill={SERIES(4)} isAnimationActive={false} />
        </ComposedChart>
      </Cell>
      <Cell title={t("panel.charts.survival")} rows={edgeSurvivalRows(summaries)} cols={["step", "retained", "removedByGv", "removedByCcm", "removedByPgo"]}>
        <ComposedChart width={170} height={88} data={edgeSurvivalRows(summaries)} onMouseMove={onMove}>
          <XAxis dataKey="step" interval={0} tick={tick} tickLine={false} axisLine={false} />
          {activeRef}
          <Bar dataKey="retained" stackId="a" fill={SERIES(5)} isAnimationActive={false} />
          <Bar dataKey="removedByGv" stackId="a" fill={SERIES(6)} isAnimationActive={false} />
          <Bar dataKey="removedByCcm" stackId="a" fill={SERIES(7)} isAnimationActive={false} />
          <Bar dataKey="removedByPgo" stackId="a" fill={SERIES(8)} isAnimationActive={false} />
        </ComposedChart>
      </Cell>
      <Cell title={t("panel.charts.duration")} rows={durationRows(summaries)} cols={["step", "duration_s"]}>
        <ComposedChart width={170} height={88} data={durationRows(summaries)} onMouseMove={onMove}>
          <XAxis dataKey="step" interval={0} tick={tick} tickLine={false} axisLine={false} />
          {activeRef}
          <Bar dataKey="duration_s" fill={SERIES(8)} isAnimationActive={false} />
        </ComposedChart>
      </Cell>
      <Cell title={t("panel.charts.newCulled")} rows={newCulledRows(summaries)} cols={["step", "num_new", "num_culled"]}>
        <LineChart width={170} height={88} data={newCulledRows(summaries)} onMouseMove={onMove}>
          <XAxis dataKey="step" interval={0} tick={tick} tickLine={false} axisLine={false} />
          {activeRef}
          <Line dataKey="num_new" stroke={SERIES(1)} dot={false} isAnimationActive={false} />
          <Line dataKey="num_culled" stroke={SERIES(2)} dot={false} isAnimationActive={false} />
        </LineChart>
      </Cell>
      <Cell title={t("panel.charts.ate")} rows={ateRows(summaries)} cols={["step", "ate_trans_rmse", "ate_rot_rmse"]}>
        {ateRows(summaries).every((r) => r.ate_trans_rmse === null) ? (
          <div className="flex h-[88px] items-center justify-center text-muted-foreground">{t("panel.charts.ateNone")}</div>
        ) : (
          <LineChart width={170} height={88} data={ateRows(summaries)} onMouseMove={onMove}>
            <XAxis dataKey="step" interval={0} tick={tick} tickLine={false} axisLine={false} />
            {activeRef}
            <Line dataKey="ate_trans_rmse" stroke={SERIES(1)} dot={false} isAnimationActive={false} connectNulls />
            <Line dataKey="ate_rot_rmse" stroke={SERIES(2)} dot={false} isAnimationActive={false} connectNulls />
          </LineChart>
        )}
      </Cell>
    </div>
  );
}
