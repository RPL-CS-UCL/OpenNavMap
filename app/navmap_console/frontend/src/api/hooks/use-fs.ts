import { useQuery } from "@tanstack/react-query";
import { apiGet } from "../client";
import { qk } from "../query-keys";
import { fsListingSchema, fsRootsSchema } from "../schemas";

export function useFsRoots() {
  return useQuery({ queryKey: qk.fs.roots, queryFn: () => apiGet("/api/fs/roots", fsRootsSchema) });
}

export function useFsList(path: string | null) {
  return useQuery({
    queryKey: qk.fs.list(path ?? ""),
    queryFn: () => apiGet(`/api/fs/list?path=${encodeURIComponent(path ?? "")}`, fsListingSchema),
    enabled: path !== null,
  });
}
