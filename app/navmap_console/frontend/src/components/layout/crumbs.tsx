import { useRegion } from "@/api/hooks/use-regions";
import { useSession } from "@/api/hooks/use-sessions";

// Breadcrumb labels that need data: fall back to the raw id until it loads.
export function RegionCrumb({ rid }: { rid: string }) {
  const { data } = useRegion(rid);
  return <>{data?.name ?? rid}</>;
}

export function SessionCrumb({ rid, sid }: { rid: string; sid: string }) {
  const { data } = useSession(rid, sid);
  return <>{data?.name ?? sid}</>;
}
