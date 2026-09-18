import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { z } from "zod";
import { apiGet, apiSend, apiUpload } from "../client";
import { qk } from "../query-keys";
import { sessionSchema } from "../schemas";
import type { Session } from "../types";

const base = (rid: string) => `/api/regions/${rid}/sessions`;

export function useSessions(rid: string) {
  return useQuery({ queryKey: qk.sessions.list(rid), queryFn: () => apiGet(base(rid), z.array(sessionSchema)) });
}

export function useSession(rid: string, sid: string) {
  return useQuery({
    queryKey: qk.sessions.one(rid, sid),
    queryFn: () => apiGet(`${base(rid)}/${sid}`, sessionSchema),
  });
}

function useInvalidateSessions(rid: string) {
  const qc = useQueryClient();
  return () => {
    void qc.invalidateQueries({ queryKey: qk.sessions.list(rid) });
    void qc.invalidateQueries({ queryKey: qk.regions.one(rid) });
    void qc.invalidateQueries({ queryKey: qk.regions.all });
  };
}

export function useRegisterSession(rid: string) {
  const invalidate = useInvalidateSessions(rid);
  return useMutation({
    mutationFn: (body: { path: string; name?: string }) =>
      apiSend<Session>("POST", `${base(rid)}/register`, body, sessionSchema),
    onSuccess: invalidate,
  });
}

export function useUploadSession(rid: string) {
  const invalidate = useInvalidateSessions(rid);
  return useMutation({
    mutationFn: (form: FormData) => apiUpload<Session>(`${base(rid)}/upload`, form, sessionSchema),
    onSuccess: invalidate,
  });
}

export function useDeleteSession(rid: string) {
  const invalidate = useInvalidateSessions(rid);
  return useMutation({
    mutationFn: (sid: string) => apiSend<void>("DELETE", `${base(rid)}/${sid}`),
    onSuccess: invalidate,
  });
}

export function useRevalidateSession(rid: string) {
  const qc = useQueryClient();
  const invalidate = useInvalidateSessions(rid);
  return useMutation({
    mutationFn: (sid: string) => apiSend<Session>("POST", `${base(rid)}/${sid}/validate`, undefined, sessionSchema),
    onSuccess: (session) => {
      qc.setQueryData(qk.sessions.one(rid, session.id), session);
      invalidate();
    },
  });
}
