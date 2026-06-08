# OpenNavMap / LiteVLoc Repo Split Design

**Date:** 2026-06-08
**Branch:** tro_opennavmap_third_version
**Status:** Approved

---

## 1. Background and Goal

The current private repository originated from the LiteVLoc codebase but now carries the full OpenNavMap system. OpenNavMap is the newly proposed academic system for multi-session topometric mapping and scalable image-goal navigation. LiteVLoc is its visual localization submodule.

To make the academic narrative, code boundaries, and public release path clear, this design separates OpenNavMap and LiteVLoc into two distinct repositories.

**OpenNavMap** is the main system repository, responsible for:
- Multi-session topometric map construction
- Submap loading, alignment, and merging
- Cross-session loop creation
- Map-level pose graph optimization
- Multi-session mapping experiments

**LiteVLoc** is an independently runnable ROS/Python visual localization package, responsible for:
- Offline and online visual localization
- Goal-image navigation runtime
- Pose fusion runtime
- Map-free localization and relative pose estimation benchmarks

OpenNavMap uses LiteVLoc as a git submodule. LiteVLoc does not depend on OpenNavMap.

---

## 2. Repository and Branch Strategy

### OpenNavMap

| Field | Value |
|---|---|
| Current repo | `RPL-CS-UCL/litevloc_private` |
| Target repo | `RPL-CS-UCL/opennavmap` |
| Migration method | GitHub repo rename (preserves history, issues, branches) |
| Work branch | `tro_opennavmap_third_version` |

### LiteVLoc

| Field | Value |
|---|---|
| Repo | `RPL-CS-UCL/litevloc_code` |
| Protected branch | `main` (must NOT be overwritten) |
| Integration branch | `opennavmap-integration` |
| Submodule pin | Fixed to a specific commit on `opennavmap-integration` |

The new LiteVLoc code will be pushed only to `opennavmap-integration`. The existing `main` branch at `litevloc_code` is preserved untouched.

---

## 3. Repository Relationship

```
RPL-CS-UCL/opennavmap
└── third_party/litevloc_code   ->   RPL-CS-UCL/litevloc_code @ <pinned commit on opennavmap-integration>
```

Dependency direction is strictly one-way:

```
OpenNavMap -> LiteVLoc
LiteVLoc  -/-> OpenNavMap
```

Specifically:
- OpenNavMap map merging core does **not** depend on LiteVLoc submodule.
- The following OpenNavMap integration entry points **may** reference LiteVLoc submodule indirectly via `PYTHONPATH`: online navigation launch files, demo scripts, and paper experiment scripts that invoke the full localization + navigation stack.
- The following OpenNavMap files must **never** import from `third_party/litevloc_code`: `map_merge_pipeline.py`, `map_manager.py`, `benchmark_mms/`, `benchmark_vpr/`, `benchmark_kf_selection/`, `utils/utils_map_merging.py`, `utils/gtsam_pose_graph.py`, `utils/gen_covis_trav_edges.py`.
- LiteVLoc must **never** import from OpenNavMap.

---

## 4. File Ownership

### 4.1 OpenNavMap

OpenNavMap retains:

```
python/map_merge_pipeline.py
python/map_manager.py
python/benchmark_mms/
python/benchmark_vpr/
python/benchmark_kf_selection/
python/utils/utils_map_merging.py
python/utils/gtsam_pose_graph.py          # kept here intentionally; see §6
python/utils/utils_geom.py                # kept here intentionally; see §6
python/utils/utils_image.py               # kept here intentionally; see §6
python/utils/gen_covis_trav_edges.py
scripts/run_map_merging.sh
scripts/run_map_merging_ablation_studies.sh
scripts/run_benchmark_vpr_submission.sh
scripts/run_benchmark_vpr_evaluation.sh
scripts/run_benchmark_kf_selection.sh
scripts/run_benchmark_kf_submission.sh
scripts/run_benchmark_kf_evaluation.sh
paper_writing/
docs/
```

OpenNavMap does **not** carry as primary scope:
```
python/segment_change/        # remove from main scope; archive or delete later
python/ltl_task_planner/      # remove from main scope; archive or delete later
python/utils/benchmark/       # LiteVLoc-owned; will be removed in Phase C Batch 3
```

### 4.2 LiteVLoc

LiteVLoc integration branch carries:

**Core localization**
```
python/loc_pipeline.py
python/ros_loc_pipeline.py
python/global_planner.py
python/ros_global_planner.py
python/pose_fusion.py
python/ros_pose_fusion.py
python/ros_publish_graph.py
python/ros_publish_goal_image.py
python/depth_registration.py
python/camera_keyframe_select.py
```

**Map representation (shared, owned by LiteVLoc)**
```
python/image_graph.py
python/point_graph.py
python/image_node.py
python/point_node.py
```

**Benchmarks**
```
python/benchmark_map_free/
python/benchmark_rpe/
```

**Utilities**
```
python/utils/utils_vpr_method.py
python/utils/utils_image_matching_method.py
python/utils/pose_solver.py
python/utils/pose_solver_default.py
python/utils/utils_geom.py               # also retained in OpenNavMap; see §6
python/utils/utils_image.py              # also retained in OpenNavMap; see §6
python/utils/utils_pipeline.py
python/utils/utils_shortest_path.py
python/utils/utils_ros/
python/utils/benchmark/
python/test/
```

**Launch files**
```
launch/run_vloc_online_anymal.launch
launch/run_vloc_online_simuenv.launch
launch/run_vloc_offline_files.launch
launch/run_pose_fusion.launch
launch/run_navigation_interface_simuenv.launch    # requires external navigation_interface; optional
launch/run_depth_reg.launch
```

**Scripts**
```
scripts/run_loc_pipeline.sh
scripts/record_rosbag_loc_simu.sh
scripts/export_odom_vloc_simu.sh
scripts/export_odom_vloc_anymal.sh
scripts/run_benchmark_mf_submission.sh
scripts/run_benchmark_mf_evaluation.sh
scripts/run_benchmark_rpe_submission.sh
scripts/run_benchmark_rpe_evaluation.sh
scripts/run_benchmark_rpe_depth_generation.sh
scripts/run_finetune_rpe_test.sh
```

**Scripts: final assignment to be resolved in Phase B step 1 by reading each file**

Decision rule: if the script's primary invocation calls `map_merge_pipeline.py`, `benchmark_vpr/`, or `benchmark_kf_selection/`, it goes to OpenNavMap; otherwise it goes to LiteVLoc.

Preliminary assignments based on content inspection:

```
scripts/run_batch_gendata.sh           -> OpenNavMap  (generates multi-session map data for map merging)
scripts/run_batch_gendata_vpr.sh       -> OpenNavMap  (generates VPR benchmark data for benchmark_vpr)
scripts/run_ego_blur.sh                -> LiteVLoc    (image preprocessing for localization datasets)
scripts/run_batch_extract_vpr_iqa.sh   -> LiteVLoc    (extracts VPR descriptors and IQA from map sequences, calls rosrun litevloc)
scripts/run_batch_vpr_seq_slam.sh      -> LiteVLoc    (runs sequential VPR matching, calls rosrun litevloc)
```

These assignments must be confirmed by reading each script at the start of Phase B before copying files.

---

## 5. Import and PYTHONPATH Rules

Short-term strategy: avoid large-scale package restructuring. Use explicit `PYTHONPATH` ordering instead.

```bash
export OPENNAVMAP_ROOT=/path/to/opennavmap
export PYTHONPATH=$OPENNAVMAP_ROOT/python:$OPENNAVMAP_ROOT/third_party/litevloc_code/python:$PYTHONPATH
```

Priority order:
1. OpenNavMap `python/` first — ensures `map_merge_pipeline.py` resolves the local `utils/gtsam_pose_graph.py`
2. LiteVLoc submodule `python/` second — allows integration demos to resolve LiteVLoc runtime modules

**Forbidden imports:**
- `map_merge_pipeline.py` must not import from `third_party/litevloc_code`
- `map_manager.py` must not import from `third_party/litevloc_code`
- `benchmark_mms/`, `benchmark_vpr/`, `benchmark_kf_selection/` must not import from `third_party/litevloc_code`
- Any LiteVLoc module must not import from the OpenNavMap repo

The three shared utils (`gtsam_pose_graph.py`, `utils_geom.py`, `utils_image.py`) are resolved via `PYTHONPATH` priority: OpenNavMap's `python/` is placed first, so its local copies shadow the LiteVLoc submodule copies when running OpenNavMap.

---

## 6. Shared Utility Duplication Rule

The following utility files are intentionally duplicated across both repositories:

```
python/utils/gtsam_pose_graph.py
python/utils/utils_geom.py
python/utils/utils_image.py
```

This duplication is limited to these three files only and must not be generalised to other utilities.

**OpenNavMap** retains its own copies because:
- `map_merge_pipeline.py` imports `utils.gtsam_pose_graph.PoseGraph`, `utils.utils_geom`, and `utils.utils_image`
- `map_manager.py` imports `utils.utils_geom`
- OpenNavMap map merging core must not depend on the LiteVLoc submodule

**LiteVLoc** retains its own copies because:
- `pose_fusion.py` imports `utils.gtsam_pose_graph.PoseGraph`
- `loc_pipeline.py`, `ros_loc_pipeline.py`, `image_graph.py`, `point_graph.py` import `utils.utils_geom` and/or `utils.utils_image`
- LiteVLoc must be independently runnable without any dependency on OpenNavMap

The duplication is intentional and acceptable. These are low-level geometry/image/GTSAM utilities, not core research logic. If divergence becomes significant in the future, extracting a common package can be reconsidered at that point.

---

## 7. Submodule Configuration

`.gitmodules` entry in OpenNavMap:

```ini
[submodule "third_party/litevloc_code"]
    path = third_party/litevloc_code
    url = git@github.com:RPL-CS-UCL/litevloc_code.git
    branch = opennavmap-integration
```

The submodule must be pinned to a specific commit, not rely on floating branch state. This ensures:
- Paper experiments are reproducible
- A specific OpenNavMap commit corresponds to a known LiteVLoc version
- Future LiteVLoc integration branch updates do not silently break OpenNavMap

---

## 8. Migration Execution Order

### Phase A: OpenNavMap branch and remote

1. Confirm working state:
   ```bash
   git status && git branch -vv && git remote -v
   ```

2. Create new work branch from current `tro_opennavmap_second_version`:
   ```bash
   git checkout tro_opennavmap_second_version
   git checkout -b tro_opennavmap_third_version
   git push -u origin tro_opennavmap_third_version
   ```

3. Rename GitHub repo via GitHub UI:
   ```
   RPL-CS-UCL/litevloc_private -> RPL-CS-UCL/opennavmap
   ```

4. Update local remote:
   ```bash
   git remote set-url origin git@github.com:RPL-CS-UCL/opennavmap.git
   git remote -v
   ```

### Phase B: LiteVLoc integration branch

**Convention: no machine-specific absolute paths are used. All commands rely on environment variables.**

```bash
export OPENNAVMAP_ROOT=$(git rev-parse --show-toplevel)   # set from inside OpenNavMap repo
export LITEVLOC_WORKDIR=/tmp/litevloc_code                # temporary clone location
```

1. Confirm script assignments for the five ambiguous scripts by reading each one:
   ```bash
   head -20 "$OPENNAVMAP_ROOT/scripts/run_batch_gendata.sh"
   head -20 "$OPENNAVMAP_ROOT/scripts/run_batch_gendata_vpr.sh"
   head -20 "$OPENNAVMAP_ROOT/scripts/run_ego_blur.sh"
   head -20 "$OPENNAVMAP_ROOT/scripts/run_batch_extract_vpr_iqa.sh"
   head -20 "$OPENNAVMAP_ROOT/scripts/run_batch_vpr_seq_slam.sh"
   ```
   Apply the decision rule in §4.2 and update the assignment list if preliminary assignments are wrong.

2. Clone LiteVLoc public repo separately (do not touch the OpenNavMap working tree):
   ```bash
   git clone git@github.com:RPL-CS-UCL/litevloc_code.git "$LITEVLOC_WORKDIR"
   cd "$LITEVLOC_WORKDIR"
   git checkout main
   git checkout -b opennavmap-integration
   ```

3. Copy LiteVLoc-owned files from the OpenNavMap working tree into `$LITEVLOC_WORKDIR`, preserving directory structure. Also copy metadata files.

   ```bash
   # Python modules
   for f in loc_pipeline ros_loc_pipeline global_planner ros_global_planner \
             pose_fusion ros_pose_fusion ros_publish_graph ros_publish_goal_image \
             depth_registration camera_keyframe_select \
             image_graph point_graph image_node point_node; do
     cp "$OPENNAVMAP_ROOT/python/${f}.py" "$LITEVLOC_WORKDIR/python/"
   done

   # Benchmark dirs and test
   cp -r "$OPENNAVMAP_ROOT/python/benchmark_map_free" "$LITEVLOC_WORKDIR/python/"
   cp -r "$OPENNAVMAP_ROOT/python/benchmark_rpe"      "$LITEVLOC_WORKDIR/python/"
   cp -r "$OPENNAVMAP_ROOT/python/test"                "$LITEVLOC_WORKDIR/python/"
   if [ -d "$OPENNAVMAP_ROOT/python/config" ]; then
     cp -r "$OPENNAVMAP_ROOT/python/config" "$LITEVLOC_WORKDIR/python/"
   fi

   # Utils — whitelist copy: only LiteVLoc-owned files; avoids pulling OpenNavMap-only utils
   mkdir -p "$LITEVLOC_WORKDIR/python/utils"
   for f in utils_vpr_method utils_image_matching_method \
             pose_solver pose_solver_default \
             utils_geom utils_image gtsam_pose_graph \
             utils_pipeline utils_shortest_path \
             utils_stamped_poses utils_convert_pose_format utils_convert_pose_to_kml \
             utils_viz2d_camera utils_viz3d_camera utils_viz2d_graph \
             utils_setting_color_font utils_trajectory \
             extract_vpr_descriptors extract_iqa vpr_sequence_matching \
             vpr_sequence_matching_adaptive vpr_single_matching vpr_topological_filter \
             vpr_graph_search base_graph base_node; do
     [ -f "$OPENNAVMAP_ROOT/python/utils/${f}.py" ] && \
       cp "$OPENNAVMAP_ROOT/python/utils/${f}.py" "$LITEVLOC_WORKDIR/python/utils/"
   done
   # Copy __init__ if present
   [ -f "$OPENNAVMAP_ROOT/python/utils/__init__.py" ] && \
     cp "$OPENNAVMAP_ROOT/python/utils/__init__.py" "$LITEVLOC_WORKDIR/python/utils/"
   # Subdirectories owned by LiteVLoc
   cp -r "$OPENNAVMAP_ROOT/python/utils/utils_ros"   "$LITEVLOC_WORKDIR/python/utils/"
   cp -r "$OPENNAVMAP_ROOT/python/utils/benchmark"   "$LITEVLOC_WORKDIR/python/utils/"

   # Launch files
   cp -r "$OPENNAVMAP_ROOT/launch" "$LITEVLOC_WORKDIR/"

   # Scripts — whitelist copy: only LiteVLoc-owned scripts
   mkdir -p "$LITEVLOC_WORKDIR/scripts"
   for s in run_loc_pipeline record_rosbag_loc_simu \
             export_odom_vloc_simu export_odom_vloc_anymal \
             run_benchmark_mf_submission run_benchmark_mf_evaluation \
             run_benchmark_rpe_submission run_benchmark_rpe_evaluation \
             run_benchmark_rpe_depth_generation run_finetune_rpe_test \
             run_ego_blur run_batch_extract_vpr_iqa run_batch_vpr_seq_slam; do
     [ -f "$OPENNAVMAP_ROOT/scripts/${s}.sh" ] && \
       cp "$OPENNAVMAP_ROOT/scripts/${s}.sh" "$LITEVLOC_WORKDIR/scripts/"
   done
   # Note: run_batch_gendata.sh, run_batch_gendata_vpr.sh go to OpenNavMap — not copied here

   # Metadata
   cp "$OPENNAVMAP_ROOT/requirements.txt" "$LITEVLOC_WORKDIR/"
   cp "$OPENNAVMAP_ROOT/environment.yaml" "$LITEVLOC_WORKDIR/"
   cp "$OPENNAVMAP_ROOT/package.xml"      "$LITEVLOC_WORKDIR/"  # ROS package name stays litevloc
   cp "$OPENNAVMAP_ROOT/CMakeLists.txt"   "$LITEVLOC_WORKDIR/"
   cp "$OPENNAVMAP_ROOT/README.md"        "$LITEVLOC_WORKDIR/"  # update afterwards
   cp "$OPENNAVMAP_ROOT/.gitignore"       "$LITEVLOC_WORKDIR/"
   ```

4. Generate and run the LiteVLoc validation script **before any git operations**:

   ```bash
   mkdir -p "$LITEVLOC_WORKDIR/tmp/split_validation"
   cat > "$LITEVLOC_WORKDIR/tmp/split_validation/validate_litevloc_integration_imports.py" << 'EOF'
   import os, pathlib, importlib, sys

   litevloc_root = pathlib.Path(os.environ["LITEVLOC_WORKDIR"]).resolve()
   opennavmap_root = pathlib.Path(os.environ["OPENNAVMAP_ROOT"]).resolve()

   required_modules = [
       "loc_pipeline", "ros_loc_pipeline",
       "image_graph", "point_graph", "image_node", "point_node",
       "pose_fusion", "global_planner",
       "benchmark_map_free", "benchmark_rpe",
   ]
   util_modules = ["utils.utils_geom", "utils.utils_image", "utils.gtsam_pose_graph"]

   for name in required_modules + util_modules:
       mod = importlib.import_module(name)
       mod_file = pathlib.Path(mod.__file__).resolve()
       assert litevloc_root in mod_file.parents, \
           f"FAIL {name}: {mod_file} not under {litevloc_root}"
       assert opennavmap_root not in mod_file.parents, \
           f"FAIL {name}: {mod_file} leaks from OpenNavMap root"
       print(f"OK {name}: {mod_file}")

   print("\nAll LiteVLoc imports resolved correctly.")
   EOF

   PYTHONPATH="$LITEVLOC_WORKDIR/python" \
   LITEVLOC_WORKDIR="$LITEVLOC_WORKDIR" \
   OPENNAVMAP_ROOT="$OPENNAVMAP_ROOT" \
   python "$LITEVLOC_WORKDIR/tmp/split_validation/validate_litevloc_integration_imports.py"
   ```

   If any assertion fails, fix the copy step before proceeding. Do NOT commit until all assertions pass.

5. Run existing LiteVLoc tests:
   ```bash
   cd "$LITEVLOC_WORKDIR"
   PYTHONPATH="$LITEVLOC_WORKDIR/python" \
   python -m pytest python/test/test_pose_solver.py -v
   ```

6. Push integration branch:
   ```bash
   cd "$LITEVLOC_WORKDIR"
   git add -A
   git commit -m "sync: import LiteVLoc modules from OpenNavMap working tree"
   git push -u origin opennavmap-integration
   export LITEVLOC_INTEGRATION_COMMIT=$(git rev-parse HEAD)
   echo "LiteVLoc integration commit: $LITEVLOC_INTEGRATION_COMMIT"
   ```

### Phase C: OpenNavMap submodule and cleanup

**Convention: run all commands from within the OpenNavMap repo root.**

```bash
export OPENNAVMAP_ROOT=$(git rev-parse --show-toplevel)
```

1. Add LiteVLoc as submodule in OpenNavMap:
   ```bash
   cd "$OPENNAVMAP_ROOT"
   git submodule add -b opennavmap-integration \
     git@github.com:RPL-CS-UCL/litevloc_code.git \
     third_party/litevloc_code
   ```

2. Pin the submodule to the specific commit from Phase B step 6:
   ```bash
   cd "$OPENNAVMAP_ROOT/third_party/litevloc_code"
   git checkout "$LITEVLOC_INTEGRATION_COMMIT"
   cd "$OPENNAVMAP_ROOT"
   git add third_party/litevloc_code
   git commit -m "chore: pin LiteVLoc submodule to opennavmap-integration commit"
   ```
   If `$LITEVLOC_INTEGRATION_COMMIT` is not in shell, retrieve it with:
   ```bash
   cd "$LITEVLOC_WORKDIR" && git rev-parse HEAD
   ```

3. Update OpenNavMap documentation and CLAUDE.md to reflect OpenNavMap identity and new boundary.

4. Generate and run the OpenNavMap core validation script **before any rm operations**.

   The script must be run with **only** OpenNavMap's own `python/` in `PYTHONPATH` — deliberately excluding `third_party/litevloc_code/python` — to verify that map merging core is fully standalone:

   ```bash
   mkdir -p "$OPENNAVMAP_ROOT/tmp/opennavmap_split_validation"
   cat > "$OPENNAVMAP_ROOT/tmp/opennavmap_split_validation/validate_opennavmap_core_imports.py" << 'EOF'
   import os, pathlib, importlib, sys

   opennavmap_root = pathlib.Path(os.environ["OPENNAVMAP_ROOT"]).resolve()
   expected_utils = opennavmap_root / "python" / "utils"

   # Verify core OpenNavMap modules import cleanly
   core_modules = [
       "map_merge_pipeline", "map_manager",
       "benchmark_mms",                    # import top-level __init__ or a submodule
   ]
   # Spot-check benchmark dirs that must not import from third_party
   benchmark_spot = ["benchmark_vpr.evaluation", "benchmark_kf_selection.keyframe_selection"]
   shared_utils = ["utils.gtsam_pose_graph", "utils.utils_geom", "utils.utils_image"]

   for name in core_modules + benchmark_spot + shared_utils:
       mod = importlib.import_module(name)
       mod_file = pathlib.Path(mod.__file__).resolve()
       assert opennavmap_root in mod_file.parents, \
           f"FAIL {name}: {mod_file} not under OpenNavMap root"
       assert "third_party" not in str(mod_file), \
           f"FAIL {name}: {mod_file} resolved from third_party (should be local)"
       print(f"OK {name}: {mod_file}")

   print("\nAll OpenNavMap core imports resolved locally — no LiteVLoc dependency.")
   EOF

   # Run with ONLY OpenNavMap local python/ — no LiteVLoc submodule in PYTHONPATH
   PYTHONPATH="$OPENNAVMAP_ROOT/python" \
   OPENNAVMAP_ROOT="$OPENNAVMAP_ROOT" \
   python "$OPENNAVMAP_ROOT/tmp/opennavmap_split_validation/validate_opennavmap_core_imports.py"
   ```

   If any assertion fails, stop and fix the file ownership before proceeding to cleanup.

5. Run existing OpenNavMap tests:
   ```bash
   PYTHONPATH="$OPENNAVMAP_ROOT/python" \
   python -m pytest "$OPENNAVMAP_ROOT/python/test/test_shortest_path.py" -v
   ```
   Note: only run tests for files that are OpenNavMap-retained. Skip tests for LiteVLoc-owned modules.

6. **Gate: only after steps 4–5 pass**, remove LiteVLoc-owned duplicate files from OpenNavMap in small batches. Before each batch, run a reference scan to confirm no retained file still imports the files being deleted:

   **Batch 1 — runtime python modules:**
   ```bash
   # Reference scan before deletion
   rg "loc_pipeline|global_planner|pose_fusion|depth_registration|camera_keyframe_select" \
     "$OPENNAVMAP_ROOT/python/map_merge_pipeline.py" \
     "$OPENNAVMAP_ROOT/python/map_manager.py" \
     "$OPENNAVMAP_ROOT/python/benchmark_mms" \
     "$OPENNAVMAP_ROOT/python/benchmark_vpr" \
     "$OPENNAVMAP_ROOT/python/benchmark_kf_selection" \
     "$OPENNAVMAP_ROOT/python/utils/utils_map_merging.py" \
     "$OPENNAVMAP_ROOT/python/utils/gtsam_pose_graph.py" \
     "$OPENNAVMAP_ROOT/python/utils/utils_geom.py" \
     "$OPENNAVMAP_ROOT/python/utils/utils_image.py" \
     "$OPENNAVMAP_ROOT/python/utils/gen_covis_trav_edges.py" || true
   # If matches found, stop and fix imports before deleting.

   rm -f "$OPENNAVMAP_ROOT/python/loc_pipeline.py"
   rm -f "$OPENNAVMAP_ROOT/python/ros_loc_pipeline.py"
   rm -f "$OPENNAVMAP_ROOT/python/global_planner.py"
   rm -f "$OPENNAVMAP_ROOT/python/ros_global_planner.py"
   rm -f "$OPENNAVMAP_ROOT/python/pose_fusion.py"
   rm -f "$OPENNAVMAP_ROOT/python/ros_pose_fusion.py"
   rm -f "$OPENNAVMAP_ROOT/python/ros_publish_graph.py"
   rm -f "$OPENNAVMAP_ROOT/python/ros_publish_goal_image.py"
   rm -f "$OPENNAVMAP_ROOT/python/depth_registration.py"
   rm -f "$OPENNAVMAP_ROOT/python/camera_keyframe_select.py"
   rm -f "$OPENNAVMAP_ROOT/python/image_graph.py"
   rm -f "$OPENNAVMAP_ROOT/python/point_graph.py"
   rm -f "$OPENNAVMAP_ROOT/python/image_node.py"
   rm -f "$OPENNAVMAP_ROOT/python/point_node.py"

   # Re-run validation after batch 1
   PYTHONPATH="$OPENNAVMAP_ROOT/python" \
   OPENNAVMAP_ROOT="$OPENNAVMAP_ROOT" \
   python "$OPENNAVMAP_ROOT/tmp/opennavmap_split_validation/validate_opennavmap_core_imports.py"
   ```

   **Batch 2 — benchmark dirs:**
   ```bash
   # Reference scan
   rg "benchmark_map_free|benchmark_rpe" \
     "$OPENNAVMAP_ROOT/python/map_merge_pipeline.py" \
     "$OPENNAVMAP_ROOT/python/map_manager.py" \
     "$OPENNAVMAP_ROOT/python/benchmark_mms" \
     "$OPENNAVMAP_ROOT/python/benchmark_vpr" \
     "$OPENNAVMAP_ROOT/python/benchmark_kf_selection" || true

   rm -rf "$OPENNAVMAP_ROOT/python/benchmark_map_free"
   rm -rf "$OPENNAVMAP_ROOT/python/benchmark_rpe"

   # Re-run validation after batch 2
   PYTHONPATH="$OPENNAVMAP_ROOT/python" \
   OPENNAVMAP_ROOT="$OPENNAVMAP_ROOT" \
   python "$OPENNAVMAP_ROOT/tmp/opennavmap_split_validation/validate_opennavmap_core_imports.py"
   ```

   **Batch 3 — LiteVLoc-only utils (excluding the three shared utils):**
   ```bash
   # Reference scan
   rg "utils_vpr_method|utils_image_matching_method|pose_solver|utils_pipeline|utils_shortest_path|utils_ros" \
     "$OPENNAVMAP_ROOT/python/map_merge_pipeline.py" \
     "$OPENNAVMAP_ROOT/python/map_manager.py" \
     "$OPENNAVMAP_ROOT/python/benchmark_mms" \
     "$OPENNAVMAP_ROOT/python/benchmark_vpr" \
     "$OPENNAVMAP_ROOT/python/benchmark_kf_selection" || true

   rm -f "$OPENNAVMAP_ROOT/python/utils/utils_vpr_method.py"
   rm -f "$OPENNAVMAP_ROOT/python/utils/utils_image_matching_method.py"
   rm -f "$OPENNAVMAP_ROOT/python/utils/pose_solver.py"
   rm -f "$OPENNAVMAP_ROOT/python/utils/pose_solver_default.py"
   rm -f "$OPENNAVMAP_ROOT/python/utils/utils_pipeline.py"
   rm -f "$OPENNAVMAP_ROOT/python/utils/utils_shortest_path.py"
   rm -rf "$OPENNAVMAP_ROOT/python/utils/utils_ros"
   rm -rf "$OPENNAVMAP_ROOT/python/utils/benchmark"

   # DO NOT remove: gtsam_pose_graph.py, utils_geom.py, utils_image.py — see §6
   # DO NOT remove: utils_map_merging.py, gen_covis_trav_edges.py — OpenNavMap-owned

   # Re-run validation after batch 3
   PYTHONPATH="$OPENNAVMAP_ROOT/python" \
   OPENNAVMAP_ROOT="$OPENNAVMAP_ROOT" \
   python "$OPENNAVMAP_ROOT/tmp/opennavmap_split_validation/validate_opennavmap_core_imports.py"
   ```

   **Batch 4a — confirmed LiteVLoc-owned scripts:**
   ```bash
   # Scripts and launch files do not affect Python imports, but verify no OpenNavMap
   # retained Python file sources or calls them before deleting.
   rg "run_loc_pipeline|record_rosbag|export_odom_vloc|benchmark_mf|benchmark_rpe|run_finetune_rpe" \
     "$OPENNAVMAP_ROOT/python/map_merge_pipeline.py" \
     "$OPENNAVMAP_ROOT/python/map_manager.py" \
     "$OPENNAVMAP_ROOT/python/benchmark_mms" \
     "$OPENNAVMAP_ROOT/python/benchmark_vpr" \
     "$OPENNAVMAP_ROOT/python/benchmark_kf_selection" || true

   rm -f "$OPENNAVMAP_ROOT/scripts/run_loc_pipeline.sh"
   rm -f "$OPENNAVMAP_ROOT/scripts/record_rosbag_loc_simu.sh"
   rm -f "$OPENNAVMAP_ROOT/scripts/export_odom_vloc_simu.sh"
   rm -f "$OPENNAVMAP_ROOT/scripts/export_odom_vloc_anymal.sh"
   rm -f "$OPENNAVMAP_ROOT/scripts/run_benchmark_mf_submission.sh"
   rm -f "$OPENNAVMAP_ROOT/scripts/run_benchmark_mf_evaluation.sh"
   rm -f "$OPENNAVMAP_ROOT/scripts/run_benchmark_rpe_submission.sh"
   rm -f "$OPENNAVMAP_ROOT/scripts/run_benchmark_rpe_evaluation.sh"
   rm -f "$OPENNAVMAP_ROOT/scripts/run_benchmark_rpe_depth_generation.sh"
   rm -f "$OPENNAVMAP_ROOT/scripts/run_finetune_rpe_test.sh"
   rm -f "$OPENNAVMAP_ROOT/launch/run_vloc_online_anymal.launch"
   rm -f "$OPENNAVMAP_ROOT/launch/run_vloc_online_simuenv.launch"
   rm -f "$OPENNAVMAP_ROOT/launch/run_vloc_offline_files.launch"
   rm -f "$OPENNAVMAP_ROOT/launch/run_pose_fusion.launch"
   rm -f "$OPENNAVMAP_ROOT/launch/run_navigation_interface_simuenv.launch"
   rm -f "$OPENNAVMAP_ROOT/launch/run_depth_reg.launch"

   # Re-run validation after batch 4a
   PYTHONPATH="$OPENNAVMAP_ROOT/python" \
   OPENNAVMAP_ROOT="$OPENNAVMAP_ROOT" \
   python "$OPENNAVMAP_ROOT/tmp/opennavmap_split_validation/validate_opennavmap_core_imports.py"
   ```

   **Batch 4b — scripts confirmed LiteVLoc-owned during Phase B step 1:**
   Only execute after confirming assignment in §4.2:
   ```bash
   rm -f "$OPENNAVMAP_ROOT/scripts/run_ego_blur.sh"
   rm -f "$OPENNAVMAP_ROOT/scripts/run_batch_extract_vpr_iqa.sh"
   rm -f "$OPENNAVMAP_ROOT/scripts/run_batch_vpr_seq_slam.sh"
   ```

---

## 9. Recommended Commit Sequence

### OpenNavMap

```
docs: add OpenNavMap/LiteVLoc repo split design
chore: create tro_opennavmap_third_version branch
docs: rename repository identity to OpenNavMap
chore: add LiteVLoc submodule at pinned commit
docs: update CLAUDE.md and README for OpenNavMap identity
cleanup: remove LiteVLoc-owned runtime modules from OpenNavMap scope
```

### LiteVLoc

```
chore: create opennavmap-integration branch
sync: import updated LiteVLoc localization core
sync: add navigation and pose fusion runtime modules
sync: add benchmark_map_free and benchmark_rpe
docs: document OpenNavMap integration branch usage
```

---

## 10. Validation Checklist

Temporary validation scripts are generated during migration and not committed to git (`tmp/` is gitignored).

### LiteVLoc integration branch

Run with isolated PYTHONPATH (`PYTHONPATH="$LITEVLOC_WORKDIR/python"` only):

- [ ] `validate_litevloc_integration_imports.py` passes all assertions
- [ ] All required module `__file__` paths are under `$LITEVLOC_WORKDIR`
- [ ] No module `__file__` path is under `$OPENNAVMAP_ROOT`
- [ ] `python python/loc_pipeline.py --help` succeeds
- [ ] `python python/global_planner.py --help` succeeds
- [ ] `python python/pose_fusion.py --help` succeeds
- [ ] `python -m pytest python/test/test_pose_solver.py` passes
- [ ] `python -m pytest python/test/test_batch_image_matching_method.py` passes
- [ ] `python -m pytest python/test/test_batch_vpr_method.py` passes
- [ ] `benchmark_map_free` and `benchmark_rpe` imports resolve

### OpenNavMap

Run with standalone PYTHONPATH (`PYTHONPATH="$OPENNAVMAP_ROOT/python"` only — no LiteVLoc submodule):

- [ ] `validate_opennavmap_core_imports.py` passes all assertions
- [ ] `map_merge_pipeline`, `map_manager` `__file__` paths are under `$OPENNAVMAP_ROOT/python`
- [ ] `utils.gtsam_pose_graph`, `utils.utils_geom`, `utils.utils_image` `__file__` paths are under `$OPENNAVMAP_ROOT/python/utils` (not `third_party/`)
- [ ] `python python/map_merge_pipeline.py --help` succeeds
- [ ] `benchmark_mms/`, `benchmark_vpr/`, `benchmark_kf_selection/` imports resolve
- [ ] LiteVLoc submodule exists at `third_party/litevloc_code` and is pinned to a specific commit (not floating branch)
- [ ] README.md identifies repo as OpenNavMap
- [ ] CLAUDE.md identifies repo as OpenNavMap
- [ ] `docs/repo_structure_brief.md` reflects new boundary

---

## 11. Known Risks and Mitigations

| Risk | Mitigation |
|---|---|
| `sys.path` order causes wrong `utils/` to be imported | Enforce PYTHONPATH order: OpenNavMap local first |
| Duplicate `gtsam_pose_graph.py`, `utils_geom.py`, `utils_image.py` diverge over time | Accepted intentionally; limited to these three files only; revisit if divergence becomes significant |
| Scripts contain hard-coded `/Titan/.../litevloc` paths | Fix paths in a dedicated commit after boundary stabilises |
| ROS package name `litevloc` conflicts with OpenNavMap naming | Keep `litevloc` as the ROS package name in LiteVLoc repo; OpenNavMap can rename its `package.xml` separately later |
| Floating submodule branch breaks reproducibility | Always pin submodule to a specific commit |
| `navigation_interface` external dependency may confuse LiteVLoc users | Document as optional in LiteVLoc README |
| Deleting duplicate files prematurely breaks imports | Validate via temporary scripts before any `rm`; delete in small batches; re-validate after each batch |
| `segment_change` and `ltl_task_planner` left as dead code | Remove from docs scope in Phase C; physical deletion is a separate decision |
| Transitive dependencies of retained utils not checked | Before each deletion batch, run reference scan (`rg`) over all OpenNavMap-retained files |
| Machine-specific absolute paths make spec non-portable | No absolute paths in spec; all commands use `$OPENNAVMAP_ROOT` and `$LITEVLOC_WORKDIR` |
