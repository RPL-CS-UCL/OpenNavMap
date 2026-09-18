import { HttpResponse, http } from "msw";
import type { Region, Session } from "@/api/types";
import { region, session, sessionInvalid } from "./fixtures";

// In-memory state shared by all handlers; call resetState() between tests.
let regions: Region[] = [];
let sessions: Session[] = [];
let counter = 0;

export function resetState(): void {
  regions = [structuredClone(region)];
  sessions = [structuredClone(session), structuredClone(sessionInvalid)];
  counter = 0;
}
resetState();

function withCounts(r: Region): Region {
  return { ...r, session_count: sessions.filter((s) => s.region_id === r.id).length, run_count: 0 };
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

  http.get("/api/regions", () => HttpResponse.json(regions.map(withCounts))),
  http.post("/api/regions", async ({ request }) => {
    const body = (await request.json()) as Partial<Region>;
    counter += 1;
    const created: Region = {
      id: `reg_new_${counter}`,
      name: body.name ?? "",
      description: body.description ?? "",
      vpr: body.vpr ?? { method: "cosplace", backbone: "ResNet18", dim: 256 },
      image_size: body.image_size ?? [512, 288],
      created_at: new Date().toISOString(),
      head: null,
    };
    regions.push(created);
    return HttpResponse.json(withCounts(created), { status: 201 });
  }),
  http.get("/api/regions/:rid", ({ params }) => {
    const r = regions.find((x) => x.id === params.rid);
    return r ? HttpResponse.json(withCounts(r)) : notFound("region");
  }),
  http.delete("/api/regions/:rid", ({ params }) => {
    regions = regions.filter((x) => x.id !== params.rid);
    return new HttpResponse(null, { status: 204 });
  }),

  http.get("/api/regions/:rid/sessions", ({ params }) =>
    HttpResponse.json(sessions.filter((s) => s.region_id === params.rid)),
  ),
  http.post("/api/regions/:rid/sessions/register", async ({ params, request }) => {
    const body = (await request.json()) as { path: string; name?: string };
    if (!body.path.startsWith("/")) return HttpResponse.json({ detail: "outside allowed roots" }, { status: 400 });
    counter += 1;
    const created: Session = {
      ...structuredClone(session),
      id: `ses_new_${counter}`,
      region_id: String(params.rid),
      name: body.name ?? body.path.split("/").filter(Boolean).pop() ?? "",
      path: body.path,
      source: "path",
    };
    sessions.push(created);
    return HttpResponse.json(created, { status: 201 });
  }),
  http.post("/api/regions/:rid/sessions/upload", async ({ params, request }) => {
    const form = await request.formData();
    const file = form.get("file");
    if (!(file instanceof File) || !/\.(zip|7z)$/i.test(file.name)) {
      return HttpResponse.json({ detail: "unsupported archive type" }, { status: 400 });
    }
    counter += 1;
    const created: Session = {
      ...structuredClone(session),
      id: `ses_new_${counter}`,
      region_id: String(params.rid),
      name: String(form.get("name") ?? "") || file.name.replace(/\.(zip|7z)$/i, ""),
      path: `/tmp/console/regions/${String(params.rid)}/sessions/ses_new_${counter}/data`,
      source: "upload",
    };
    sessions.push(created);
    return HttpResponse.json(created, { status: 201 });
  }),
  http.get("/api/regions/:rid/sessions/:sid", ({ params }) => {
    const s = sessions.find((x) => x.id === params.sid && x.region_id === params.rid);
    return s ? HttpResponse.json(s) : notFound("session");
  }),
  http.post("/api/regions/:rid/sessions/:sid/validate", ({ params }) => {
    const s = sessions.find((x) => x.id === params.sid);
    return s ? HttpResponse.json(s) : notFound("session");
  }),
  http.delete("/api/regions/:rid/sessions/:sid", ({ params }) => {
    sessions = sessions.filter((x) => x.id !== params.sid);
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
];
