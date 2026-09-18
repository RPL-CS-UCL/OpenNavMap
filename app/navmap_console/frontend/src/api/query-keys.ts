export const qk = {
  health: ["health"] as const,
  regions: { all: ["regions"] as const, one: (rid: string) => ["regions", rid] as const },
  sessions: {
    list: (rid: string) => ["regions", rid, "sessions"] as const,
    one: (rid: string, sid: string) => ["regions", rid, "sessions", sid] as const,
  },
  fs: { roots: ["fs", "roots"] as const, list: (path: string) => ["fs", "list", path] as const },
  jobs: {
    all: ["jobs"] as const,
    list: (status: string) => ["jobs", "list", status] as const,
    one: (jid: string) => ["jobs", jid] as const,
    log: (jid: string) => ["jobs", jid, "log"] as const,
  },
  runs: {
    list: (rid: string) => ["regions", rid, "runs"] as const,
    one: (rid: string, runId: string) => ["regions", rid, "runs", runId] as const,
  },
  params: { merge: ["params", "merge"] as const },
};
