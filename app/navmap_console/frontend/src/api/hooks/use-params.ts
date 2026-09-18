import { useQuery } from "@tanstack/react-query";
import { z } from "zod";
import { apiGet } from "../client";
import { qk } from "../query-keys";
import { paramSpecSchema } from "../schemas";

export function useMergeParams() {
  return useQuery({
    queryKey: qk.params.merge,
    queryFn: () => apiGet("/api/params/merge", z.array(paramSpecSchema)),
    staleTime: Infinity,
  });
}
