import type { Region, Session, ValidationReport } from "@/api/types";

export const region: Region = {
  id: "reg_1",
  name: "campus",
  description: "UCL campus, Aria glasses",
  vpr: { method: "cosplace", backbone: "ResNet18", dim: 256 },
  image_size: [512, 288],
  created_at: "2026-09-18T10:00:00+00:00",
  head: null,
  session_count: 1,
  run_count: 0,
};

const validReport: ValidationReport = {
  ok: true,
  num_frames: 12,
  has_gt: false,
  has_gps: false,
  has_iqa: true,
  descriptor_dim: 256,
  files: [
    { name: "poses.txt", required: true, status: "ok", lines: 12, detail: "" },
    { name: "timestamps.txt", required: true, status: "ok", lines: 12, detail: "" },
    { name: "intrinsics.txt", required: true, status: "ok", lines: 12, detail: "" },
    { name: "gps_data.txt", required: true, status: "ok", lines: 12, detail: "all zero" },
    { name: "database_descriptors.txt", required: true, status: "ok", lines: 12, detail: "dim 256" },
    { name: "poses_abs_gt.txt", required: false, status: "missing", lines: null, detail: "" },
    { name: "iqa_data.txt", required: false, status: "ok", lines: 12, detail: "" },
    { name: "edges_covis.txt", required: false, status: "ok", lines: 17, detail: "" },
    { name: "edges_odom.txt", required: false, status: "ok", lines: 11, detail: "" },
    { name: "edges_trav.txt", required: false, status: "ok", lines: 11, detail: "" },
  ],
  warnings: [],
  errors: [],
  checked_at: "2026-09-18T10:05:00+00:00",
};

export const session: Session = {
  id: "ses_1",
  region_id: "reg_1",
  name: "000",
  kind: "submap",
  source: "path",
  path: "/Titan/dataset/example/000",
  num_frames: 12,
  has_gt: false,
  has_gps: false,
  has_iqa: true,
  validation: validReport,
  created_at: "2026-09-18T10:05:00+00:00",
};

export const sessionInvalid: Session = {
  ...session,
  id: "ses_2",
  name: "broken",
  path: "/Titan/dataset/example/broken",
  has_iqa: false,
  validation: {
    ...validReport,
    ok: false,
    has_iqa: false,
    files: validReport.files.map((f) =>
      f.name === "gps_data.txt" ? { ...f, status: "missing", lines: null } : f,
    ),
    errors: ["gps_data.txt is required but missing"],
    warnings: ["edges_covis.txt is 3 lines short; those frames are dropped silently by the loader"],
  },
};
