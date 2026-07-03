# Spec: OpenNavMap Map Merge Rerun Visualization

**Date:** 2026-07-03  
**Status:** Approved design, pending implementation plan

---

## Background

OpenNavMap map merging currently saves intermediate result directories such as:

- `/Titan/dataset/data_opennavmap/map_multisession_eval/ucl_campus_aria/s00000_results_in_kf_spgo_cc_seqmatch`
- `/Titan/dataset/data_opennavmap/map_multisession_eval/ucl_campus_aria/s00000_results_r4_kf_spgo_cc_seqmatch`
- `/Titan/dataset/data_opennavmap/map_multisession_eval/vineyard/s00000_results_in_kf_spgo_cc_seqmatch`

The requested visualization should produce an offline Rerun `.rrd` file that can replay the map-merging process over time. The replay should show submaps appearing one by one, then moving through VPR, geometric verification, metric localization, keyframe culling, pose graph optimization, and final map update.

The first implementation should support both:

1. future map-merging runs that record complete process events while running, then save a `.rrd` for offline viewing;
2. existing completed result directories, converted in read-only mode without recomputing descriptors, VPR, geometric verification, or pose estimation.

---

## Goals

1. Add a professional Rerun visualization for incremental map merging.
2. Use a map-dominant layout: the 3D/topometric map is the primary view, while D-matrix and image evidence are supporting views.
3. Use a keyframe-level replay timeline where possible.
4. Keep visualization logic isolated from map-merging algorithm logic.
5. Support read-only replay of existing results without recomputation.
6. Use the vineyard result directory as the first validation scene.
7. Keep existing map-merging behavior unchanged when Rerun visualization is disabled.

---

## Non-Goals

1. Do not provide real-time Rerun viewing while map merging runs. The output is an offline `.rrd` file.
2. Do not recompute missing details for existing result directories.
3. Do not add an extra dashboard or web UI.
4. Do not make Rerun views perform interactive algorithm recomputation.
5. Do not fake unavailable per-keyframe data when a read-only result directory lacks it.

---

## Chosen Approach

Use a hybrid event-log architecture:

1. `MapMergeVizRecorder` records structured events from future `map_merge_pipeline.py` runs.
2. `MapMergeRerunWriter` converts structured events into Rerun `.rrd` files.
3. `MapMergeResultReplay` reads existing result directories and reconstructs a best-effort event stream.

This is preferred over directly writing Rerun calls inside `map_merge_pipeline.py` because it keeps the core algorithm clean and lets online recording and offline replay share one output format.

---

## Proposed Modules

```text
python/visualization/map_merge_viz_events.py
  Defines event dataclasses or typed dictionaries, stage names, and artifact refs.
  This module must not import rerun and should not depend on map_merge_pipeline classes.

python/visualization/map_merge_viz_recorder.py
  Runtime recorder used by map_merge_pipeline.
  It converts in-memory graph, pose, match, culling, and PGO state into structured events.
  It also saves image artifacts in configured formats.

python/visualization/map_merge_rerun_writer.py
  The only module that imports rerun.
  It owns blueprints, entity paths, timelines, color rules, axis scaling, and .rrd output.

python/visualization/map_merge_result_replay.py
  Read-only parser for existing result directories.
  It reads merge_* directories, poses, edges, preds artifacts, G2O files, and keyframe culling images.

python/visualization/map_merge_result_to_rerun.py
  CLI entry point for converting an existing result directory to .rrd.
```

---

## Data Flow

Future complete recording:

```text
map_merge_pipeline
  -> MapMergeVizRecorder
  -> structured events + artifact refs
  -> MapMergeRerunWriter
  -> map_merge_process.rrd
```

Existing read-only replay:

```text
result_dir
  -> MapMergeResultReplay
  -> best-effort structured events + saved artifact refs
  -> MapMergeRerunWriter
  -> replay.rrd
```

---

## Event Model

Events are append-only records. Each event describes one algorithm fact and leaves rendering decisions to `MapMergeRerunWriter`.

Required common fields:

```text
event_type
merge_step
stage
submap_id
keyframe_id
timestamp_optional
payload
artifact_refs
```

Event types:

```text
submap_loaded
descriptors_ready
dmatrix_ready
vpr_candidate
sequence_match_result
gv_result
metric_localization_result
keyframe_culling_result
pose_graph_before_pgo
pose_graph_after_pgo
submap_merged
```

`payload` stores lightweight structured data such as node ids, query/reference ids, scores, inlier counts, confidence, poses, edge endpoints, and PGO error values.

`artifact_refs` stores paths to heavier artifacts such as query/reference JPEG images, D-matrix PNG images, matching JPEG images, culling evidence images, and G2O files.

`sequence_match_result` represents the temporal VPR output when the configured matcher uses sequence matching or graph/DP search over the D-matrix. It should include the query row, reference row, path or point sequence when available, score, and D-matrix PNG artifact reference. If the matcher produces only independent candidates, `vpr_candidate` is sufficient.

`gv_result` represents geometric verification for a candidate. Its payload must include the database node id, query node id, retained/rejected status, inlier count, score or confidence when available, and optional matching image artifact reference. `retained` maps to green visualization; `rejected` maps to gray visualization.

---

## Runtime Recording Integration Points

`map_merge_pipeline.py` should only call the recorder at explicit process boundaries. It should not directly import or call `rerun`.

Integration points:

```text
run_incremental_merge:
  start_recording
  submap_loaded

perform_global_loc:
  descriptors_ready
  dmatrix_ready
  vpr_candidate per keyframe
  sequence_match_result when using sequence or DP matching
  gv_result per candidate

perform_local_loc:
  metric_localization_result per refined edge

perform_keyframe_culling:
  keyframe_culling_result

merge_single_submap:
  pose_graph_before_pgo
  pose_graph_after_pgo
  submap_merged
```

When `--rerun-viz` is disabled, no recorder should be created and existing behavior must remain unchanged.

---

## Read-Only Replay Strategy

Existing result conversion must not rerun descriptors, VPR matching, geometric verification, or pose estimation. It should parse saved artifacts only.

Recoverable from existing result directories:

- merge step order from chained `merge_*` directories;
- map poses from `poses.txt`;
- odometry, covisibility, and traversability edges from `edges_odom.txt`, `edges_covis.txt`, and `edges_trav.txt`;
- stage artifacts from `preds/*.jpg`, `preds/*.png`, and `preds/*.pdf`;
- pose graph state from `initial_pose_graph.g2o` and `refine_pose_graph.g2o`;
- culling evidence from `preds/kf_vis/*` and `cull_node_info.txt` if present.

Potentially missing in old results:

- raw per-keyframe D-matrix values;
- exact sequence matching path when only rendered stage images were saved;
- per-candidate inlier keypoints;
- exact rejected edge history;
- metric localization in-memory confidence and landmark gain.

When details are missing, the `.rrd` should show an English status note, for example:

```text
GV per-keyframe details unavailable in saved result; showing saved stage artifact.
Sequence matching path unavailable in saved result; showing saved D-matrix artifact.
Only read-only replay is enabled; no recomputation was performed.
```

---

## Rerun Blueprint

The default Rerun layout is map-dominant.

Primary view:

```text
Spatial3DView: Map Merge Process
origin: /world
```

This view shows:

- `/world` coordinate axes;
- final/reference map nodes and edges;
- current submap nodes, camera frustums, axes, and odometry edges;
- VPR candidate edges;
- GV retained and rejected edges;
- metric localization edges;
- keyframe culling and replacement evidence;
- PGO before/after graph and motion vectors.

Supporting views:

```text
Spatial2DView: Difference Matrix
Spatial2DView: Query / Reference Pair
TextLogView or TextDocument: Stage Summary
```

The map view should remain the dominant screen area. D-matrix, query/reference image, matching image, and status text are evidence panels.

---

## Entity Path Convention

```text
/world/axes

/world/final_map/nodes/{node_id}
/world/final_map/edges/odom
/world/final_map/edges/covis
/world/final_map/edges/trav

/world/current_submap/{submap_id}/nodes/{node_id}
/world/current_submap/{submap_id}/edges/odom
/world/current_submap/{submap_id}/cameras/{node_id}

/world/matches/vpr_candidates
/world/matches/gv_inliers
/world/matches/rejected_gv
/world/matches/rejected_ccm
/world/matches/metric_edges

/world/culling/culled_nodes
/world/culling/replacement_edges

/world/pgo/before/nodes
/world/pgo/before/edges
/world/pgo/after/nodes
/world/pgo/after/edges
/world/pgo/motion_vectors

/evidence/dmatrix
/evidence/query_image
/evidence/reference_image
/evidence/matching_image
/evidence/keyframe_culling

/status/stage_summary
/status/statistics
```

---

## Timelines

Use `keyframe_id` as the main replay timeline when keyframe-level data is available.

Additional timelines:

```text
merge_step: the current submap merge step
stage: load_submap, descriptor, vpr, sequence_matching, gv, metric_loc, culling, pgo_before, pgo_after, merged
```

Stage-level events without a natural keyframe should use a synthetic keyframe slot around the current merge step. The exact synthetic keyframe rule should be deterministic and documented in code.

---

## Visual Encoding

All visible text in Rerun should default to English, including stage names, labels, warnings, summaries, and overlays.

Color rules:

| Item | Color rule |
|---|---|
| final/reference map | blue-gray |
| current submap | orange |
| VPR candidate | yellow |
| GV retained | green |
| GV rejected | gray |
| CCM rejected | red or muted red |
| metric localization edge | green |
| PGO before | translucent |
| PGO after | solid |
| culled keyframe | red marker or translucent node |

`/world` must include a visible coordinate frame. Axis length and radius must scale with scene size because some scenes are large.

Recommended auto scale:

```text
axis_length = max(scene_extent * 0.08, 2.0)
axis_radius = max(scene_extent * 0.001, 0.03)
```

Camera axes and frustums should also scale from scene extent so they remain visible in UCL campus or vineyard-sized maps.

---

## Artifact Format And Size Control

D-matrix evidence should use PNG because it is matrix-like visual evidence and benefits from lossless encoding.

Camera, query, reference, and matching images may use JPEG to control `.rrd` size.

Configurable defaults:

```text
--rerun-image-format jpg
--rerun-jpeg-quality 85
--rerun-dmatrix-format png
--rerun-max-match-images-per-step 200
--rerun-axis-scale auto
```

Default size controls:

1. Do not store every map node RGB image by default.
2. Store current query/reference images and selected matching evidence.
3. Store D-matrix as normalized PNG image evidence.
4. Limit matching images per merge step.
5. Prefer structured 3D primitives for map nodes, edges, axes, and pose graph state.

---

## Command Line Interface

Future full recording from map merging:

```bash
python python/map_merge_pipeline.py \
  ...existing args... \
  --rerun-viz \
  --rerun-output /path/to/map_merge_process.rrd \
  --rerun-image-format jpg \
  --rerun-jpeg-quality 85 \
  --rerun-dmatrix-format png \
  --rerun-max-match-images-per-step 200 \
  --rerun-axis-scale auto
```

Read-only conversion from an existing result directory:

```bash
python python/visualization/map_merge_result_to_rerun.py \
  --result-dir /Titan/dataset/data_opennavmap/map_multisession_eval/vineyard/s00000_results_in_kf_spgo_cc_seqmatch \
  --rerun-output /path/to/vineyard_map_merge_process.rrd \
  --mode readonly \
  --rerun-image-format jpg \
  --rerun-jpeg-quality 85 \
  --rerun-dmatrix-format png \
  --rerun-axis-scale auto
```

Both runtime recording and read-only conversion should share the Rerun-specific option names: `--rerun-output`, `--rerun-image-format`, `--rerun-jpeg-quality`, `--rerun-dmatrix-format`, and `--rerun-axis-scale`.

`scripts/run_map_merging.sh` should only add optional forwarding for these flags. Default script behavior should not change.

---

## Error Handling

Visualization should not break normal map merging unless the user explicitly requested Rerun output and the required dependency is missing.

Rules:

1. If `--rerun-viz` is disabled, no Rerun-related behavior runs.
2. If `--rerun-viz` is enabled but `rerun` is unavailable, fail clearly because output was explicitly requested.
3. If a single image or artifact is missing, write a warning and continue.
4. If read-only replay lacks a stage artifact, emit an English status note and continue.
5. If G2O parsing fails, fall back to `poses.txt` plus `edges_*.txt`.
6. If auto axis scaling fails, use a conservative default and warn.

---

## Vineyard Validation Scene

Use this existing result directory as the first validation scene:

```text
/Titan/dataset/data_opennavmap/map_multisession_eval/vineyard/s00000_results_in_kf_spgo_cc_seqmatch
```

Read-only exploration confirmed it contains:

- 5 merge steps: `merge_0` through `merge_0_1_2_3_4`;
- per-step `poses.txt`, `timestamps.txt`, `intrinsics.txt`, `edges_odom.txt`, `edges_covis.txt`, and `edges_trav.txt`;
- per-step `initial_pose_graph.g2o`;
- `refine_pose_graph.g2o` for merge steps after the first;
- D-matrix and pose graph artifacts under `preds/`;
- keyframe culling evidence under `preds/kf_vis/`.

This scene is small enough for fast iteration and covers the main read-only replay branches.

---

## Testing Strategy

Pure parser and writer tests should not require opening the Rerun viewer.

Suggested tests:

```text
test_parse_merge_chain_vineyard_like_names
test_load_mapfree_poses_and_edges
test_parse_g2o_vertices_edges
test_collect_stage_artifacts
test_readonly_replay_marks_missing_details
test_axis_scale_from_scene_extent
```

Manual validation:

1. Convert the vineyard result directory to `.rrd` in read-only mode.
2. Open the `.rrd` in Rerun.
3. Confirm that the main 3D map view shows incremental map growth.
4. Confirm that `/world/axes` is visible and scaled appropriately.
5. Confirm that D-matrix evidence appears as PNG.
6. Confirm that GV retained items are green and rejected items are gray when such events are available.
7. Confirm that all labels, warnings, and status text are English.
8. Confirm that missing read-only details produce warnings instead of recomputation or fabricated data.

---

## Acceptance Criteria

1. Running the read-only converter on the vineyard result directory generates a `.rrd` file.
2. The `.rrd` shows map growth across merge steps with a map-dominant layout.
3. The `/world` coordinate axes are visible in large scenes.
4. The replay shows available `poses`, `edges`, pose graph artifacts, D-matrix artifacts, and keyframe culling evidence.
5. Future `--rerun-viz` runtime recording supports keyframe-level VPR, GV, metric localization, culling, PGO, and merged stages.
6. D-matrix evidence is PNG; normal image evidence can be JPEG.
7. Read-only replay never recomputes missing algorithm details.
8. Existing map-merging behavior is unchanged when visualization is disabled.
