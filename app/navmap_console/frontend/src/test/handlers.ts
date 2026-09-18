import { HttpResponse, http } from "msw";
import type { Job, Region, Run, Session, StepRecord } from "@/api/types";
import { job, paramSpecs, region, run, session, sessionInvalid, steps } from "./fixtures";

// In-memory state shared by all handlers; call resetState() between tests.
// Exported as one object so tests can tweak fixtures in place (e.g. state.runs[0].status = ...).
interface HandlerState {
  regions: Region[];
  sessions: Session[];
  jobs: Job[];
  runs: Run[];
  runSteps: Record<string, StepRecord[]>;
  counter: number;
}
export const state: HandlerState = { regions: [], sessions: [], jobs: [], runs: [], runSteps: {}, counter: 0 };

export function resetState(): void {
  state.regions = [structuredClone(region)];
  state.sessions = [structuredClone(session), structuredClone(sessionInvalid)];
  state.jobs = [structuredClone(job)];
  state.runs = [structuredClone(run)];
  state.runSteps = { [run.id]: structuredClone(steps) };
  state.counter = 0;
}
resetState();

function withCounts(r: Region): Region {
  return {
    ...r,
    session_count: state.sessions.filter((s) => s.region_id === r.id).length,
    run_count: state.runs.filter((x) => x.region_id === r.id).length,
  };
}

const notFound = (what: string) => HttpResponse.json({ detail: `${what} not found` }, { status: 404 });

export const handlers = [
  http.get("/api/health", () =>
    HttpResponse.json({
      version: "0.1.0",
      data_root: "/tmp/console",
      repo_root: "/tmp/repo",
      queues: { gpu: 0, cpu: 0 },
      gpu: null,
      disk: { total: 1, free: 1 },
      cpu_list: null,
    }),
  ),

  http.get("/api/regions", () => HttpResponse.json(state.regions.map(withCounts))),
  http.post("/api/regions", async ({ request }) => {
    const body = (await request.json()) as Partial<Region>;
    state.counter += 1;
    const created: Region = {
      id: `reg_new_${state.counter}`,
      name: body.name ?? "",
      description: body.description ?? "",
      vpr: body.vpr ?? { method: "cosplace", backbone: "ResNet18", dim: 256 },
      image_size: body.image_size ?? [512, 288],
      created_at: new Date().toISOString(),
      head: null,
    };
    state.regions.push(created);
    return HttpResponse.json(withCounts(created), { status: 201 });
  }),
  http.get("/api/regions/:rid", ({ params }) => {
    const r = state.regions.find((x) => x.id === params.rid);
    return r ? HttpResponse.json(withCounts(r)) : notFound("region");
  }),
  http.delete("/api/regions/:rid", ({ params }) => {
    state.regions = state.regions.filter((x) => x.id !== params.rid);
    return new HttpResponse(null, { status: 204 });
  }),

  http.get("/api/regions/:rid/sessions", ({ params }) =>
    HttpResponse.json(state.sessions.filter((s) => s.region_id === params.rid)),
  ),
  http.post("/api/regions/:rid/sessions/register", async ({ params, request }) => {
    const body = (await request.json()) as { path: string; name?: string };
    if (!body.path.startsWith("/")) return HttpResponse.json({ detail: "outside allowed roots" }, { status: 400 });
    state.counter += 1;
    const created: Session = {
      ...structuredClone(session),
      id: `ses_new_${state.counter}`,
      region_id: String(params.rid),
      name: body.name ?? body.path.split("/").filter(Boolean).pop() ?? "",
      path: body.path,
      source: "path",
    };
    state.sessions.push(created);
    return HttpResponse.json(created, { status: 201 });
  }),
  http.post("/api/regions/:rid/sessions/upload", async ({ params, request }) => {
    const form = await request.formData();
    const file = form.get("file");
    if (!(file instanceof File) || !/\.(zip|7z)$/i.test(file.name)) {
      return HttpResponse.json({ detail: "unsupported archive type" }, { status: 400 });
    }
    state.counter += 1;
    const created: Session = {
      ...structuredClone(session),
      id: `ses_new_${state.counter}`,
      region_id: String(params.rid),
      name: String(form.get("name") ?? "") || file.name.replace(/\.(zip|7z)$/i, ""),
      path: `/tmp/console/regions/${String(params.rid)}/sessions/ses_new_${state.counter}/data`,
      source: "upload",
    };
    state.sessions.push(created);
    return HttpResponse.json(created, { status: 201 });
  }),
  http.get("/api/regions/:rid/sessions/:sid", ({ params }) => {
    const s = state.sessions.find((x) => x.id === params.sid && x.region_id === params.rid);
    return s ? HttpResponse.json(s) : notFound("session");
  }),
  http.post("/api/regions/:rid/sessions/:sid/validate", ({ params }) => {
    const s = state.sessions.find((x) => x.id === params.sid);
    return s ? HttpResponse.json(s) : notFound("session");
  }),
  http.delete("/api/regions/:rid/sessions/:sid", ({ params }) => {
    state.sessions = state.sessions.filter((x) => x.id !== params.sid);
    return new HttpResponse(null, { status: 204 });
  }),

  http.get("/api/fs/roots", () => HttpResponse.json({ roots: ["/Titan/dataset"] })),
  http.get("/api/fs/list", ({ request }) => {
    const path = new URL(request.url).searchParams.get("path") ?? "";
    if (!path.startsWith("/Titan/dataset")) return HttpResponse.json({ detail: "outside roots" }, { status: 400 });
    if (path === "/Titan/dataset") {
      return HttpResponse.json({
        path,
        parent: null,
        entries: [
          { name: "example", path: "/Titan/dataset/example", is_dir: true, looks_like_session: false, size: null },
          { name: "readme.txt", path: "/Titan/dataset/readme.txt", is_dir: false, looks_like_session: false, size: 12 },
        ],
      });
    }
    if (path === "/Titan/dataset/example") {
      return HttpResponse.json({
        path,
        parent: "/Titan/dataset",
        entries: [
          { name: "000", path: "/Titan/dataset/example/000", is_dir: true, looks_like_session: true, size: null },
          { name: "001", path: "/Titan/dataset/example/001", is_dir: true, looks_like_session: true, size: null },
        ],
      });
    }
    return notFound("path");
  }),

  // ---- state.jobs / state.runs / params (M3) ----
  http.get("/api/params/merge", () => HttpResponse.json(paramSpecs)),
  http.get("/api/jobs", ({ request }) => {
    const status = new URL(request.url).searchParams.get("status");
    return HttpResponse.json(status ? state.jobs.filter((j) => status.split(",").includes(j.status)) : state.jobs);
  }),
  http.get("/api/jobs/:jid", ({ params }) => {
    const j = state.jobs.find((x) => x.id === params.jid);
    return j ? HttpResponse.json(j) : notFound("job");
  }),
  http.post("/api/jobs/:jid/cancel", ({ params }) => {
    const j = state.jobs.find((x) => x.id === params.jid);
    if (!j) return notFound("job");
    j.status = "cancelled";
    return HttpResponse.json(j);
  }),
  http.get("/api/jobs/:jid/log", ({ request }) => {
    const after = Number(new URL(request.url).searchParams.get("after") ?? 0);
    const all = [
      "$ python map_merge_pipeline.py",
      "--- Merging submap 0: ses_1 ---",
      "PGO: final error: 0.456",
      "STEP_DONE index=0 sid=ses_1",
    ];
    const lines = all.slice(after);
    return HttpResponse.json({ lines, next: after + lines.length, total: all.length });
  }),
  http.get("/api/regions/:rid/runs", ({ params }) =>
    HttpResponse.json(state.runs.filter((r) => r.region_id === params.rid)),
  ),
  http.post("/api/regions/:rid/runs", async ({ params, request }) => {
    const body = (await request.json()) as {
      name?: string;
      kind: "merge" | "append";
      session_ids: string[];
      params: Record<string, unknown>;
    };
    if (state.runs.some((r) => r.region_id === params.rid && (r.status === "queued" || r.status === "running")))
      return HttpResponse.json({ detail: "region busy" }, { status: 409 });
    state.counter += 1;
    const created: Run = {
      ...structuredClone(run),
      id: `run_new_${state.counter}`,
      region_id: String(params.rid),
      name: body.name ?? "",
      kind: body.kind,
      session_ids: body.session_ids,
      params: body.params,
      status: "queued",
      job_id: `job_new_${state.counter}`,
      last_step_index: null,
    };
    state.runs.push(created);
    state.runSteps[created.id] = [];
    return HttpResponse.json(created, { status: 201 });
  }),
  http.get("/api/regions/:rid/runs/:runId", ({ params }) => {
    const r = state.runs.find((x) => x.id === params.runId);
    if (!r) return notFound("run");
    return HttpResponse.json({ run: r, steps: state.runSteps[r.id] ?? [], job: state.jobs.find((j) => j.id === r.job_id) ?? null });
  }),
  http.post("/api/regions/:rid/runs/:runId/cancel", ({ params }) => {
    const r = state.runs.find((x) => x.id === params.runId);
    if (!r) return notFound("run");
    r.status = "cancelled";
    return HttpResponse.json(r);
  }),
  http.post("/api/regions/:rid/map/promote", async ({ params, request }) => {
    const body = (await request.json()) as { run_id: string; step_index: number };
    const reg = state.regions.find((x) => x.id === params.rid);
    if (!reg) return notFound("region");
    reg.head = { run_id: body.run_id, step_index: body.step_index, session_ids: [], lineage: [body.run_id] };
    return HttpResponse.json(withCounts(reg));
  }),
];
