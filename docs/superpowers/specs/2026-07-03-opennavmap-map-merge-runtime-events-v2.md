# Spec: OpenNavMap Runtime Map Merge Visualization Events V2

**Date:** 2026-07-03  
**Status:** Approved for first implementation on submap0/submap1

---

## Goal

Record true runtime map-merging events while `run_map_merging.sh` / `map_merge_pipeline.py` is running. The first target is a single merge between submap0 and submap1. The output is a human-readable visualization folder that the user can inspect before any Rerun renderer is rebuilt.

---

## Scope

First version supports only online/runtime recording.

Completed result directory replay is intentionally out of scope for this version. It will be revisited after runtime event recording is validated.

---

## Output Directory

For each merge output directory, write:

```text
<output_map_path>/rerun_viz/
  metadata.json
  demo_events.jsonl
  artifacts/
    step_000/
    step_001/
```

`demo_events.jsonl` is the source of truth for the future Rerun renderer. Each line is one JSON object.

---

## Required Event Fields

Each event contains:

```json
{
  "demo_step": 0,
  "merge_step": 0,
  "stage": "submap_loaded",
  "event_type": "submap_loaded",
  "submap_id": 0,
  "keyframe_id": null,
  "payload": {},
  "artifacts": {}
}
```

`demo_step` is monotonic and is the primary timeline for the future Rerun replay.

`merge_step` is the loop index in `perform_submap_merging()`.

`keyframe_id` is optional and used when an event corresponds to a map node / image keyframe.

`payload` stores JSON-serializable scalar/list/dict data.

`artifacts` stores paths relative to `rerun_viz/`.

---

## First Runtime Events

For submap0/submap1, record at least:

```text
recording_started
submap_loaded
vio_node_observed
vio_edge_observed
descriptor_computed
dmatrix_computed
vpr_candidate
gv_candidate
metric_edge_added
pgo_before
pgo_after
map_committed
recording_finished
```

If node culling runs, also record:

```text
keyframe_culling_decision
```

---

## Runtime Hook Points

Hook only where the data is naturally produced:

- `perform_submap_merging()` loop start: `submap_loaded`, `vio_node_observed`, `vio_edge_observed`.
- `perform_global_loc()` after descriptor arrays are built: `descriptor_computed`.
- `perform_global_loc()` after `D_all` is computed: `dmatrix_computed` and a PNG artifact.
- `perform_global_loc()` when VPR returns db/query pairs: `vpr_candidate`.
- `perform_global_loc()` after geometric verification inlier check: `gv_candidate` with accepted/rejected status.
- `perform_local_loc()` after a refined metric localization edge is accepted: `metric_edge_added`.
- `perform_keyframe_culling()` results: `keyframe_culling_decision`.
- before `gtsam.writeG2o(... initial_pose_graph.g2o)`: `pgo_before`.
- after LM optimization and refined G2O write: `pgo_after`.
- after `merge_and_update_submaps()` and final edge updates: `map_committed`.

---

## Artifact Policy

The first implementation should prefer metadata and references over copying large images.

For node events, store the source image path in payload. Copying image artifacts can be added later if the renderer needs a stable packaged folder.

D-matrix should be saved as PNG in the artifacts folder because it is central to the demo and must be inspectable without rerunning the algorithm.

G2O files should be referenced as artifacts when already written by the pipeline.

---

## First Acceptance Criteria

For a submap0/submap1 runtime merge with `--rerun-viz`, the output folder must contain:

- `metadata.json`
- `demo_events.jsonl`
- at least one D-matrix PNG artifact for the second submap merge

The JSONL must include:

- two `submap_loaded` events for submap0 and submap1;
- one `vio_node_observed` event per loaded node;
- one `vio_edge_observed` event per loaded odometry edge;
- `descriptor_computed` and `dmatrix_computed` for the submap1 localization step;
- `vpr_candidate` and `gv_candidate` events;
- `metric_edge_added` events when local localization accepts edges;
- `pgo_before`, `pgo_after`, and `map_committed`.

The user will inspect this JSON before the next renderer implementation step.
