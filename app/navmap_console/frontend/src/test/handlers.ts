import { HttpResponse, http } from "msw";
import type { Job, Region, Run, Session, StepRecord, StepSummary } from "@/api/types";
import { job, paramSpecs, region, run, session, sessionInvalid, steps, summaries } from "./fixtures";
import { makeSceneFixture } from "./scene-fixture";

// In-memory state shared by all handlers; call resetState() between tests.
// Exported as one object so tests can tweak fixtures in place (e.g. state.runs[0].status = ...).
interface HandlerState {
  regions: Region[];
  sessions: Session[];
  jobs: Job[];
  runs: Run[];
  runSteps: Record<string, StepRecord[]>;
  summaries: StepSummary[];
  counter: number;
}
export const state: HandlerState = { regions: [], sessions: [], jobs: [], runs: [], runSteps: {}, summaries: [], counter: 0 };

export function resetState(): void {
  state.regions = [structuredClone(region)];
  state.sessions = [structuredClone(session), structuredClone(sessionInvalid)];
  state.jobs = [structuredClone(job)];
  state.runs = [structuredClone(run)];
  state.runSteps = { [run.id]: structuredClone(steps) };
  state.summaries = structuredClone(summaries);
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
  // ---- results (M4) ----
  http.get("/api/regions/:rid/runs/:runId/summaries", () => HttpResponse.json({ steps: state.summaries })),
  http.get("/api/regions/:rid/runs/:runId/steps/:k/scene.bin", ({ params }) => {
    const k = Number(params.k);
    if (!state.summaries.some((s) => s.index === k)) return notFound("step");
    return HttpResponse.arrayBuffer(makeSceneFixture(12 * (k + 1), 3 * k), {
      headers: { "Content-Type": "application/octet-stream" },
    });
  }),
  http.get("/api/regions/:rid/runs/:runId/steps/:k/dmatrix.json", ({ params }) => {
    if (Number(params.k) === 0) return notFound("dmatrix");
    return HttpResponse.json({
      rows: "db", cols: "query", row_node_ids: [...Array(12).keys()], col_node_ids: [...Array(12).keys()].map((i) => i + 12),
      vmin: 0.1, vmax: 0.9,
      candidates: [{ db: 0, query: 12, stage: "gv", gv_inliers: 40 }, { db: 1, query: 13, stage: "vpr", gv_inliers: 0 }],
      factors: [{ db: 0, query: 12, weight: 0.9, conf: 0.8, accepted: true, origin: "new" },
                { db: 1, query: 13, weight: 0.1, conf: 0.4, accepted: false, origin: "new" }],
    });
  }),
  http.get("/api/regions/:rid/runs/:runId/steps/:k/culling.json", ({ params }) => {
    const b = `/api/regions/${params.rid}/runs/${params.runId}`;
    return HttpResponse.json({
      culled: [{ node_id: 23, kind: "query", other: 5, prob: 0.91, method: "culled_by_forward", detail: "",
                 image_url: `${b}/nodes/23/image`, other_image_url: `${b}/nodes/5/image`, vis_url: null }],
      kept: [{ node_id: 12, kind: "query", other: 3, prob: 0.12, method: "kept", detail: "",
               image_url: `${b}/nodes/12/image`, other_image_url: `${b}/nodes/3/image`, vis_url: null }],
    });
  }),
  http.get("/api/regions/:rid/runs/:runId/steps/:k/nodes/:nid", ({ params }) => {
    const nid = Number(params.nid);
    if (nid >= 12 * (Number(params.k) + 1)) return notFound("node");
    return HttpResponse.json({
      node_id: nid, step: nid < 12 ? 0 : 1, session_id: nid < 12 ? "ses_1" : "ses_2",
      frame: `seq/${String(nid).padStart(6, "0")}.color.jpg`, timestamp: 1000 + nid,
      pos: [nid % 12 * 0.5, nid < 12 ? 0 : 2, 0], quat: [0, 0, 0, 1], pos_pre: [nid % 12 * 0.5, nid < 12 ? 0 : 2, 0],
      gt: null, degree: { odom: 2, covis: 4, trav: 2 }, flags: nid < 12 ? 0 : 1,
      image_url: `/api/regions/${params.rid}/runs/${params.runId}/nodes/${nid}/image`, cull: null,
      loops: nid === 0 ? [{ other: 12, weight: 0.9, conf: 0.5, accepted: true, origin: "new" }] : [],
    });
  }),
  http.get("/api/regions/:rid/runs/:runId/events", () => HttpResponse.json([])),
  http.post("/api/regions/:rid/runs/import", async ({ params, request }) => {
    const body = (await request.json()) as { result_dir: string; name?: string };
    if (!body.result_dir.startsWith("/")) return HttpResponse.json({ detail: "outside allowed roots" }, { status: 403 });
    state.counter += 1;
    const created: Run = {
      ...structuredClone(run), id: `run_imported_${state.counter}`, region_id: String(params.rid),
      name: body.name ?? body.result_dir.split("/").pop() ?? "", kind: "imported", status: "succeeded",
      job_id: null, num_steps_expected: 3, last_step_index: 2,
    };
    state.runs.push(created);
    return HttpResponse.json(created);
  }),
];
