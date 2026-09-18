import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { z } from "zod";
import { apiGet, apiSend } from "../client";
import { qk } from "../query-keys";
import { regionSchema } from "../schemas";
import type { Region, VprConfig } from "../types";

export interface RegionCreateBody {
  name: string;
  description?: string;
  vpr?: VprConfig;
  image_size?: number[];
}

export function useRegions() {
  return useQuery({ queryKey: qk.regions.all, queryFn: () => apiGet("/api/regions", z.array(regionSchema)) });
}

export function useRegion(rid: string) {
  return useQuery({ queryKey: qk.regions.one(rid), queryFn: () => apiGet(`/api/regions/${rid}`, regionSchema) });
}

export function useCreateRegion() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: RegionCreateBody) => apiSend<Region>("POST", "/api/regions", body, regionSchema),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.regions.all }),
  });
}

export function useDeleteRegion() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (rid: string) => apiSend<void>("DELETE", `/api/regions/${rid}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.regions.all }),
  });
}
