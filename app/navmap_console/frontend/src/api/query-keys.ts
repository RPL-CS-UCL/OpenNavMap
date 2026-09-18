export const qk = {
  health: ["health"] as const,
  regions: { all: ["regions"] as const, one: (rid: string) => ["regions", rid] as const },
  sessions: {
    list: (rid: string) => ["regions", rid, "sessions"] as const,
    one: (rid: string, sid: string) => ["regions", rid, "sessions", sid] as const,
  },
  fs: { roots: ["fs", "roots"] as const, list: (path: string) => ["fs", "list", path] as const },
};
