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
python/utils/utils_geom.py
python/utils/utils_image.py
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

---

## 6. `gtsam_pose_graph.py` Duplication Rule

`python/utils/gtsam_pose_graph.py` is intentionally duplicated across both repositories.

**OpenNavMap** retains its own copy because:
- `map_merge_pipeline.py` directly imports `utils.gtsam_pose_graph.PoseGraph`
- OpenNavMap map merging must not depend on the LiteVLoc submodule

**LiteVLoc** retains its own copy because:
- `pose_fusion.py` requires `PoseGraph`
- LiteVLoc must be independently runnable

This duplication is intentional and acceptable. `gtsam_pose_graph.py` is a GTSAM wrapper utility, not core research logic. If the two copies diverge significantly in the future, extracting a common package can be reconsidered at that point.

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

1. Confirm script assignments for the five ambiguous scripts by reading each one:
   ```bash
   cd /Titan/code/robohike_ws/src/litevloc_private
   cat scripts/run_batch_gendata.sh | head -20
   cat scripts/run_batch_gendata_vpr.sh | head -20
   cat scripts/run_ego_blur.sh | head -20
   cat scripts/run_batch_extract_vpr_iqa.sh | head -20
   cat scripts/run_batch_vpr_seq_slam.sh | head -20
   ```
   Apply the decision rule in §4.2 and update the assignment list if preliminary assignments are wrong.

2. Clone LiteVLoc public repo separately (do not touch current OpenNavMap working tree):
   ```bash
   git clone git@github.com:RPL-CS-UCL/litevloc_code.git /tmp/litevloc_code
   cd /tmp/litevloc_code
   git checkout main
   git checkout -b opennavmap-integration
   ```

3. Copy LiteVLoc-owned files from the OpenNavMap working tree into `/tmp/litevloc_code`, preserving the directory structure under `python/`, `launch/`, `scripts/`, and `docs/`. Also copy or create the following metadata files:
   - `requirements.txt` — copy from OpenNavMap (shared environment; trim OpenNavMap-only deps later)
   - `environment.yaml` — copy from OpenNavMap
   - `package.xml` — copy from OpenNavMap as-is (ROS package name stays `litevloc`)
   - `CMakeLists.txt` — copy from OpenNavMap as-is
   - `README.md` — copy from OpenNavMap, then update to describe LiteVLoc as standalone package
   - `.gitignore` — copy from OpenNavMap
   - `python/__init__.py` — copy if present
   - `python/utils/__init__.py` — copy if present
   - `python/config/` — copy if present (YACS dataset configs used by loc_pipeline)

   Example copy commands:
   ```bash
   ONAV=/Titan/code/robohike_ws/src/litevloc_private
   LVLOC=/tmp/litevloc_code

   # Python modules
   for f in loc_pipeline ros_loc_pipeline global_planner ros_global_planner \
             pose_fusion ros_pose_fusion ros_publish_graph ros_publish_goal_image \
             depth_registration camera_keyframe_select \
             image_graph point_graph image_node point_node; do
     cp $ONAV/python/${f}.py $LVLOC/python/
   done

   # Directories
   cp -r $ONAV/python/benchmark_map_free $LVLOC/python/
   cp -r $ONAV/python/benchmark_rpe      $LVLOC/python/
   cp -r $ONAV/python/utils              $LVLOC/python/
   cp -r $ONAV/python/test               $LVLOC/python/
   cp -r $ONAV/python/config             $LVLOC/python/  # if present
   cp -r $ONAV/launch                    $LVLOC/
   cp -r $ONAV/scripts                   $LVLOC/

   # Remove OpenNavMap-owned scripts from LiteVLoc copy:
   rm $LVLOC/scripts/run_map_merging.sh
   rm $LVLOC/scripts/run_map_merging_ablation_studies.sh
   rm $LVLOC/scripts/run_benchmark_vpr_submission.sh
   rm $LVLOC/scripts/run_benchmark_vpr_evaluation.sh
   rm $LVLOC/scripts/run_benchmark_kf_selection.sh
   rm $LVLOC/scripts/run_benchmark_kf_submission.sh
   rm $LVLOC/scripts/run_benchmark_kf_evaluation.sh
   rm $LVLOC/scripts/run_batch_gendata.sh
   rm $LVLOC/scripts/run_batch_gendata_vpr.sh

   # Then REMOVE utils files that belong to OpenNavMap only:
   rm $LVLOC/python/utils/utils_map_merging.py
   rm $LVLOC/python/utils/gen_covis_trav_edges.py
   # Keep gtsam_pose_graph.py (LiteVLoc needs it for pose_fusion.py)

   # Metadata
   cp $ONAV/requirements.txt $LVLOC/
   cp $ONAV/environment.yaml $LVLOC/
   cp $ONAV/package.xml      $LVLOC/
   cp $ONAV/CMakeLists.txt   $LVLOC/
   cp $ONAV/README.md        $LVLOC/
   cp $ONAV/.gitignore       $LVLOC/
   ```

4. Verify LiteVLoc runs independently. Use the full validation checklist in §10 LiteVLoc section. At minimum:
   ```bash
   cd /tmp/litevloc_code
   export PYTHONPATH=$(pwd)/python:$PYTHONPATH
   python python/loc_pipeline.py --help
   python python/global_planner.py --help
   python python/pose_fusion.py --help
   python -c "from loc_pipeline import LocPipeline; print('OK')"
   python -c "from pose_fusion import PoseFusion; print('OK')"
   python -c "from image_graph import ImageGraph; print('OK')"
   python -m pytest python/test/test_pose_solver.py -v
   ```

5. Push integration branch:
   ```bash
   git add -A
   git commit -m "sync: import LiteVLoc modules from OpenNavMap working tree"
   git push -u origin opennavmap-integration
   ```

### Phase C: OpenNavMap submodule and cleanup

1. Add LiteVLoc as submodule in OpenNavMap:
   ```bash
   git submodule add -b opennavmap-integration \
     git@github.com:RPL-CS-UCL/litevloc_code.git \
     third_party/litevloc_code
   ```

2. Pin the submodule to the specific commit pushed in Phase B:
   ```bash
   cd third_party/litevloc_code
   git checkout <commit-hash-from-phase-B-push>
   cd ../..
   git add third_party/litevloc_code
   git commit -m "chore: pin LiteVLoc submodule to opennavmap-integration commit"
   ```
   The commit hash is the output of `git rev-parse HEAD` run inside `/tmp/litevloc_code` after Phase B step 5.

3. Update OpenNavMap documentation and CLAUDE.md to reflect OpenNavMap identity and new boundary.

4. Verify map merging does not import from LiteVLoc. Use the full OpenNavMap validation checklist in §10. At minimum:
   ```bash
   export OPENNAVMAP_ROOT=$(pwd)
   export PYTHONPATH=$OPENNAVMAP_ROOT/python:$OPENNAVMAP_ROOT/third_party/litevloc_code/python:$PYTHONPATH
   python python/map_merge_pipeline.py --help
   python -c "
   import sys
   from utils.gtsam_pose_graph import PoseGraph
   # Confirm the module file is from OpenNavMap local, not LiteVLoc submodule
   import utils.gtsam_pose_graph as mod
   assert 'third_party' not in mod.__file__, f'Wrong path: {mod.__file__}'
   print('OK: gtsam_pose_graph resolved from OpenNavMap local python/')
   "
   ```

5. After validation checklist in §10 passes, remove LiteVLoc-owned duplicate files from OpenNavMap in small increments per commit. Order:
   - First remove runtime python files: `loc_pipeline.py`, `ros_loc_pipeline.py`, `global_planner.py`, etc.
   - Then remove benchmark dirs: `benchmark_map_free/`, `benchmark_rpe/`
   - Then remove LiteVLoc-only utils: `utils_vpr_method.py`, `utils_image_matching_method.py`, etc.
   - After each removal, re-run the OpenNavMap validation checklist before proceeding to the next batch.
   - Do NOT remove `gtsam_pose_graph.py` from OpenNavMap; it is intentionally retained (see §6).

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

### LiteVLoc integration branch

- [ ] `python python/loc_pipeline.py --help` succeeds
- [ ] `python python/global_planner.py --help` succeeds
- [ ] `python python/pose_fusion.py --help` succeeds
- [ ] `python -m pytest python/test/test_pose_solver.py` passes
- [ ] `python -m pytest python/test/test_batch_image_matching_method.py` passes
- [ ] `python -m pytest python/test/test_batch_vpr_method.py` passes
- [ ] benchmark_map_free and benchmark_rpe imports resolve
- [ ] No import from OpenNavMap repo

### OpenNavMap

- [ ] `python python/map_merge_pipeline.py --help` succeeds
- [ ] `map_merge_pipeline.py` imports resolve using only OpenNavMap-local `python/`
- [ ] `utils/gtsam_pose_graph.py` is resolved from OpenNavMap local path, not LiteVLoc submodule — verified by:
  ```bash
  python -c "import utils.gtsam_pose_graph as m; assert 'third_party' not in m.__file__; print('OK:', m.__file__)"
  ```
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
| Duplicate `gtsam_pose_graph.py` diverges over time | Accepted intentionally; revisit if divergence becomes significant |
| Scripts contain hard-coded `/Titan/.../litevloc` paths | Fix paths in a dedicated commit after boundary stabilizes |
| ROS package name `litevloc` conflicts with OpenNavMap naming | Keep `litevloc` as the ROS package name in LiteVLoc repo; OpenNavMap can rename its `package.xml` separately later |
| Floating submodule branch breaks reproducibility | Always pin submodule to a specific commit |
| `navigation_interface` external dependency may confuse LiteVLoc users | Document as optional in LiteVLoc README |
| Deleting duplicate files prematurely breaks imports | Validate all imports before physical deletion; delete in small increments |
| `segment_change` and `ltl_task_planner` left as dead code | Remove from docs scope in Phase C; physical deletion is a separate decision |
