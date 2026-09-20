import { type QueryClient, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { z } from "zod";
import { ApiError, apiGet, apiGetBinary, apiSend } from "../client";
import { qk } from "../query-keys";
import { type Scene, decodeSceneBundle, toScene } from "../scene-bundle";
import { cullingSchema, dmatrixSchema, nodeDetailSchema, runEventSchema, runSchema, summariesSchema } from "../schemas";
import type { Culling, DMatrix, NodeDetail, Run, RunEvent, StepSummary } from "../types";

const base = (rid: string, runId: string) => `/api/regions/${rid}/runs/${runId}`;

export const dmatrixPngUrl = (rid: string, runId: string, k: number) => `${base(rid, runId)}/steps/${k}/dmatrix.png`;
export const nodeImageUrl = (rid: string, runId: string, nid: number, w = 512) =>
  `${base(rid, runId)}/nodes/${nid}/image?w=${w}`;
export const pairImageUrl = (rid: string, runId: string, a: number, b: number, w = 384) =>
  `${base(rid, runId)}/pairs/${a}/${b}/image?w=${w}`;
export const predsFileUrl = (rid: string, runId: string, k: number, name: string) =>
  `${base(rid, runId)}/steps/${k}/preds/${name}`;

export function useStepSummaries(rid: string, runId: string) {
  return useQuery({
    queryKey: qk.results.summaries(rid, runId),
    queryFn: async (): Promise<StepSummary[]> =>
      (await apiGet(`${base(rid, runId)}/summaries`, summariesSchema)).steps,
    enabled: !!rid && !!runId,
  });
}

async function fetchScene(rid: string, runId: string, k: number): Promise<Scene> {
  return toScene(decodeSceneBundle(await apiGetBinary(`${base(rid, runId)}/steps/${k}/scene.bin`)));
}

const SCENE_GC_MS = 10 * 60 * 1000;

export function useScene(rid: string, runId: string, k: number | null) {
  return useQuery({
    queryKey: qk.results.scene(rid, runId, k ?? -1),
    queryFn: () => fetchScene(rid, runId, k as number),
    enabled: !!rid && !!runId && k !== null && k >= 0,
    staleTime: Infinity, // a finished step never changes on disk
    gcTime: SCENE_GC_MS,
  });
}

export function prefetchScene(qc: QueryClient, rid: string, runId: string, k: number): Promise<void> {
  if (k < 0) return Promise.resolve();
  return qc.prefetchQuery({
    queryKey: qk.results.scene(rid, runId, k),
    queryFn: () => fetchScene(rid, runId, k),
    staleTime: Infinity,
    gcTime: SCENE_GC_MS,
  });
}

export function useDMatrix(rid: string, runId: string, k: number | null) {
  return useQuery({
    queryKey: qk.results.dmatrix(rid, runId, k ?? -1),
    queryFn: async (): Promise<DMatrix | null> => {
      try {
        return await apiGet(`${base(rid, runId)}/steps/${k}/dmatrix.json`, dmatrixSchema);
      } catch (err) {
        if (err instanceof ApiError && err.status === 404) return null; // older results have no D_matrix.npy
        throw err;
      }
    },
    enabled: !!rid && !!runId && k !== null && k >= 0,
    staleTime: Infinity,
  });
}

export function useCulling(rid: string, runId: string, k: number | null) {
  return useQuery({
    queryKey: qk.results.culling(rid, runId, k ?? -1),
    queryFn: (): Promise<Culling> => apiGet(`${base(rid, runId)}/steps/${k}/culling.json`, cullingSchema),
    enabled: !!rid && !!runId && k !== null && k >= 0,
    staleTime: Infinity,
  });
}

export function useNodeDetail(rid: string, runId: string, k: number | null, nid: number | null) {
  return useQuery({
    queryKey: qk.results.node(rid, runId, k ?? -1, nid ?? -1),
    queryFn: (): Promise<NodeDetail> => apiGet(`${base(rid, runId)}/steps/${k}/nodes/${nid}`, nodeDetailSchema),
    enabled: !!rid && !!runId && k !== null && k >= 0 && nid !== null && nid >= 0,
    staleTime: Infinity,
  });
}

export function useEvents(rid: string, runId: string, step: number | null, types: string[] = [], opts: { refetchInterval?: number | false } = {}) {
  const typesKey = types.join(",");
  const params = new URLSearchParams();
  if (step !== null) params.set("step", String(step));
  if (typesKey) params.set("types", typesKey);
  return useQuery({
    queryKey: qk.results.events(rid, runId, step, typesKey),
    queryFn: (): Promise<RunEvent[]> => apiGet(`${base(rid, runId)}/events?${params}`, z.array(runEventSchema)),
    enabled: !!rid && !!runId,
    refetchInterval: opts.refetchInterval,
  });
}

export interface ImportRunBody {
  result_dir: string;
  sessions_root?: string;
  name?: string;
  promote?: boolean;
}

export function useImportRun(rid: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: ImportRunBody) => apiSend<Run>("POST", `/api/regions/${rid}/runs/import`, body, runSchema),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.runs.list(rid) });
      qc.invalidateQueries({ queryKey: qk.sessions.list(rid) });
      qc.invalidateQueries({ queryKey: qk.regions.one(rid) });
    },
  });
}
