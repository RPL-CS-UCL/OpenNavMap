# OpenNavMap / LiteVLoc Repo Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the current monolithic `litevloc_private` repo into two repositories — `opennavmap` (main system) and `litevloc_code` (visual localization submodule) — with a clean one-way dependency and validated file boundaries.

**Architecture:** OpenNavMap retains map merging, map-level benchmarks, paper experiments, and three shared utilities (`gtsam_pose_graph.py`, `utils_geom.py`, `utils_image.py`). LiteVLoc retains all localization runtime, navigation runtime, pose fusion, and localization benchmarks. OpenNavMap references LiteVLoc as a pinned git submodule at `third_party/litevloc_code`.

**Tech Stack:** Python 3.8, ROS (catkin), git submodules, pytest, ripgrep (`rg`), bash

**Spec:** `docs/superpowers/specs/2026-06-08-opennavmap-litevloc-repo-split-design.md`

---

## Environment Variables

All tasks assume these are set in your shell before starting. Set them once:

```bash
export OPENNAVMAP_ROOT=$(git rev-parse --show-toplevel)   # run inside the repo
export LITEVLOC_WORKDIR=/tmp/litevloc_code
```

---

## Phase A — OpenNavMap Branch and Remote

### Task 1: Create `opennavmap_third_version` branch

**Files:** git only (no file changes)

- [ ] **Step 1: Verify clean working state**

  ```bash
  git status
  git branch -vv
  git remote -v
  ```
  Expected: on `tro_opennavmap_second_version`, working tree clean, remote is `RPL-CS-UCL/litevloc_private`.

- [ ] **Step 2: Create and push new branch**

  ```bash
  git checkout tro_opennavmap_second_version
  git checkout -b opennavmap_third_version
  git push -u origin opennavmap_third_version
  ```
  Expected: new branch created and tracking `origin/opennavmap_third_version`.

- [ ] **Step 3: Verify branch**

  ```bash
  git branch -vv
  ```
  Expected: `* opennavmap_third_version` tracking `origin/opennavmap_third_version`.

---

### Task 2: Rename GitHub repo and update local remote

**Files:** git remote config only

- [ ] **Step 1: Rename repo on GitHub**

  Go to `https://github.com/RPL-CS-UCL/litevloc_private` → Settings → Repository name → rename to `opennavmap` → confirm rename.

- [ ] **Step 2: Update local remote URL**

  ```bash
  git remote set-url origin git@github.com:RPL-CS-UCL/opennavmap.git
  ```

- [ ] **Step 3: Verify remote and push work**

  ```bash
  git remote -v
  git push
  ```
  Expected: remote shows `git@github.com:RPL-CS-UCL/opennavmap.git`, push succeeds.

---

## Phase B — LiteVLoc Integration Branch

### Task 3: Confirm ambiguous script assignments

**Files:** read-only inspection of 5 scripts

- [ ] **Step 1: Read the five ambiguous scripts**

  ```bash
  head -25 "$OPENNAVMAP_ROOT/scripts/run_batch_gendata.sh"
  head -25 "$OPENNAVMAP_ROOT/scripts/run_batch_gendata_vpr.sh"
  head -25 "$OPENNAVMAP_ROOT/scripts/run_ego_blur.sh"
  head -25 "$OPENNAVMAP_ROOT/scripts/run_batch_extract_vpr_iqa.sh"
  head -25 "$OPENNAVMAP_ROOT/scripts/run_batch_vpr_seq_slam.sh"
  ```

- [ ] **Step 2: Apply decision rule**

  Decision rule (from spec §4.2): if the script's primary invocation calls `map_merge_pipeline.py`, `benchmark_vpr/`, or `benchmark_kf_selection/` → OpenNavMap; otherwise → LiteVLoc.

  Preliminary assignments (confirm or override):
  ```
  run_batch_gendata.sh        -> LiteVLoc    (external pycpptools data generation)
  run_batch_gendata_vpr.sh    -> LiteVLoc    (external pycpptools VPR data generation)
  run_ego_blur.sh             -> LiteVLoc    (image preprocessing)
  run_batch_extract_vpr_iqa.sh-> LiteVLoc    (calls rosrun litevloc)
  run_batch_vpr_seq_slam.sh   -> LiteVLoc    (calls rosrun litevloc)
  ```

  If assignments differ, update the copy list in Task 6 accordingly before executing it.

---

### Task 4: Clone LiteVLoc repo and create integration branch

**Files:** `$LITEVLOC_WORKDIR` (external clone)

- [ ] **Step 1: Clone and create branch**

  ```bash
  git clone git@github.com:RPL-CS-UCL/litevloc_code.git "$LITEVLOC_WORKDIR"
  cd "$LITEVLOC_WORKDIR"
  git checkout main
  git log --oneline -5   # confirm you see the early litevloc history
  git checkout -b opennavmap-integration
  ```
  Expected: `opennavmap-integration` branch created from `main`, `main` branch untouched.

- [ ] **Step 2: Verify main is protected**

  ```bash
  git branch -a
  ```
  Expected: `main` and `* opennavmap-integration` both visible; no local modification to `main`.

---

### Task 5: Copy LiteVLoc-owned Python modules

**Files:** `$LITEVLOC_WORKDIR/python/` (new files)

- [ ] **Step 1: Copy top-level Python modules**

  ```bash
  for f in loc_pipeline ros_loc_pipeline global_planner ros_global_planner \
            pose_fusion ros_pose_fusion ros_publish_graph ros_publish_goal_image \
            depth_registration camera_keyframe_select \
            image_graph point_graph image_node point_node; do
    cp "$OPENNAVMAP_ROOT/python/${f}.py" "$LITEVLOC_WORKDIR/python/"
  done
  ```

- [ ] **Step 2: Copy benchmark directories**

  ```bash
  cp -r "$OPENNAVMAP_ROOT/python/benchmark_map_free" "$LITEVLOC_WORKDIR/python/"
  cp -r "$OPENNAVMAP_ROOT/python/benchmark_rpe"      "$LITEVLOC_WORKDIR/python/"
  if [ -d "$OPENNAVMAP_ROOT/python/config" ]; then
    cp -r "$OPENNAVMAP_ROOT/python/config" "$LITEVLOC_WORKDIR/python/"
  fi
  ```

  Do not copy `python/test/`. Those legacy tests depend on unavailable external packages and stale hard-coded data paths. This migration uses the temporary validation script in Task 7 as its gate.

- [ ] **Step 3: Whitelist-copy utils (individual files only)**

  ```bash
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

  # __init__ if present
  [ -f "$OPENNAVMAP_ROOT/python/utils/__init__.py" ] && \
    cp "$OPENNAVMAP_ROOT/python/utils/__init__.py" "$LITEVLOC_WORKDIR/python/utils/"

  # Subdirectories fully owned by LiteVLoc
  cp -r "$OPENNAVMAP_ROOT/python/utils/utils_ros" "$LITEVLOC_WORKDIR/python/utils/"
  cp -r "$OPENNAVMAP_ROOT/python/utils/benchmark" "$LITEVLOC_WORKDIR/python/utils/"
  ```

- [ ] **Step 4: Verify no OpenNavMap-only utils were copied**

  ```bash
  ls "$LITEVLOC_WORKDIR/python/utils/"
  ```
  Confirm the following are NOT present:
  ```
  utils_map_merging.py
  gen_covis_trav_edges.py
  ```
  Confirm these ARE present:
  ```
  utils_geom.py   utils_image.py   gtsam_pose_graph.py
  ```

---

### Task 6: Copy launch, scripts, and metadata

**Files:** `$LITEVLOC_WORKDIR/launch/`, `$LITEVLOC_WORKDIR/scripts/`, metadata files

- [ ] **Step 1: Copy all launch files**

  ```bash
  cp -r "$OPENNAVMAP_ROOT/launch" "$LITEVLOC_WORKDIR/"
  ```
  All 6 launch files are LiteVLoc-owned; no deletions needed.

- [ ] **Step 2: Whitelist-copy LiteVLoc-owned scripts**

  ```bash
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
  ```
  Note: `run_batch_gendata.sh` and `run_batch_gendata_vpr.sh` were confirmed LiteVLoc-owned in Task 3 and must be copied.
  If Task 3 changed assignment for `run_ego_blur.sh`, `run_batch_extract_vpr_iqa.sh`, or `run_batch_vpr_seq_slam.sh`, adjust the list above.

- [ ] **Step 3: Copy metadata files**

  ```bash
  cp "$OPENNAVMAP_ROOT/requirements.txt" "$LITEVLOC_WORKDIR/"
  cp "$OPENNAVMAP_ROOT/environment.yaml" "$LITEVLOC_WORKDIR/"
  cp "$OPENNAVMAP_ROOT/package.xml"      "$LITEVLOC_WORKDIR/"   # ROS package name stays litevloc
  cp "$OPENNAVMAP_ROOT/CMakeLists.txt"   "$LITEVLOC_WORKDIR/"
  cp "$OPENNAVMAP_ROOT/README.md"        "$LITEVLOC_WORKDIR/"   # will be updated in Task 8
  cp "$OPENNAVMAP_ROOT/.gitignore"       "$LITEVLOC_WORKDIR/"
  [ -f "$OPENNAVMAP_ROOT/python/__init__.py" ] && \
    cp "$OPENNAVMAP_ROOT/python/__init__.py" "$LITEVLOC_WORKDIR/python/"
  ```

---

### Task 7: Generate and run LiteVLoc validation script

**Files:** `$LITEVLOC_WORKDIR/tmp/split_validation/validate_litevloc_integration_imports.py` (not committed)

- [ ] **Step 1: Generate validation script**

  ```bash
  mkdir -p "$LITEVLOC_WORKDIR/tmp/split_validation"
  cat > "$LITEVLOC_WORKDIR/tmp/split_validation/validate_litevloc_integration_imports.py" << 'EOF'
  import os, pathlib, importlib.util, sys

  litevloc_root = pathlib.Path(os.environ["LITEVLOC_WORKDIR"]).resolve()
  opennavmap_root = pathlib.Path(os.environ["OPENNAVMAP_ROOT"]).resolve()

  required_modules = [
      "loc_pipeline", "ros_loc_pipeline",
      "image_graph", "point_graph", "image_node", "point_node",
      "pose_fusion", "global_planner",
  ]
  required_dirs = ["benchmark_map_free", "benchmark_rpe"]
  util_modules = ["utils.utils_geom", "utils.utils_image", "utils.gtsam_pose_graph"]

  failed = []
  for name in required_modules + util_modules:
      try:
          spec = importlib.util.find_spec(name)
          if spec is None or spec.origin is None:
              failed.append(f"FAIL {name}: module spec not found")
              continue
          mod_file = pathlib.Path(spec.origin).resolve()
          if litevloc_root not in mod_file.parents:
              failed.append(f"FAIL {name}: {mod_file} not under {litevloc_root}")
          elif opennavmap_root in mod_file.parents:
              failed.append(f"FAIL {name}: {mod_file} leaks from OpenNavMap root")
          else:
              print(f"OK   {name}: {mod_file}")
      except Exception as e:
          failed.append(f"FAIL {name}: import error — {e}")

  for name in required_dirs:
      dir_path = litevloc_root / "python" / name
      if not dir_path.is_dir():
          failed.append(f"FAIL {name}: directory not found at {dir_path}")
      elif opennavmap_root in dir_path.resolve().parents:
          failed.append(f"FAIL {name}: directory leaks from OpenNavMap root")
      else:
          print(f"OK   {name}: {dir_path.resolve()}")

  if failed:
      print("\n--- FAILURES ---")
      for f in failed:
          print(f)
      sys.exit(1)

  print("\nAll LiteVLoc imports resolved correctly.")
  EOF
  ```

- [ ] **Step 2: Run validation with isolated PYTHONPATH**

  ```bash
  cd "$LITEVLOC_WORKDIR"
  PYTHONPATH="$LITEVLOC_WORKDIR/python" \
  LITEVLOC_WORKDIR="$LITEVLOC_WORKDIR" \
  OPENNAVMAP_ROOT="$OPENNAVMAP_ROOT" \
  python tmp/split_validation/validate_litevloc_integration_imports.py
  ```
  Expected: all lines print `OK ...`, script exits 0.
  If any `FAIL`: go back to Task 5 or 6 and fix the missing/misrouted file.

- [ ] **Step 3: Skip legacy pytest suite for this migration**

  Do not run or require `python/test/` in this task. The copied legacy tests require unavailable external packages (`pycpptools`, `faiss`, VPR model repos) and hard-coded local data. The migration acceptance gate is Step 2's isolated path validation script.

---

### Task 8: Update LiteVLoc README and commit

**Files:** `$LITEVLOC_WORKDIR/README.md`, then git commit

- [ ] **Step 1: Update README to describe LiteVLoc as standalone package**

  Open `$LITEVLOC_WORKDIR/README.md` and update the intro paragraph to say:

  ```
  LiteVLoc is an independently runnable ROS/Python visual localization package.
  It is used by OpenNavMap (https://github.com/RPL-CS-UCL/opennavmap) as a git submodule,
  but can also be used as a standalone localization system.
  ```

  Remove or update any references to `litevloc_private` or the old private repo name.

- [ ] **Step 2: Ensure tmp/ is gitignored**

  ```bash
  grep -q "^tmp/" "$LITEVLOC_WORKDIR/.gitignore" || echo "tmp/" >> "$LITEVLOC_WORKDIR/.gitignore"
  ```

- [ ] **Step 3: Ensure legacy tests are not tracked**

  ```bash
  cd "$LITEVLOC_WORKDIR"
  git rm -r --cached python/test 2>/dev/null || true
  rm -rf python/test
  ```

- [ ] **Step 4: Commit and push**

  ```bash
  cd "$LITEVLOC_WORKDIR"
  git add -A
  git status   # review what is staged; confirm tmp/ is NOT staged
  git commit -m "sync: import LiteVLoc modules from OpenNavMap working tree"
  git push -u origin opennavmap-integration
  export LITEVLOC_INTEGRATION_COMMIT=$(git rev-parse HEAD)
  echo "LiteVLoc integration commit: $LITEVLOC_INTEGRATION_COMMIT"
  # Save the commit hash to a file so it survives shell restarts
  echo "$LITEVLOC_INTEGRATION_COMMIT" > "$OPENNAVMAP_ROOT/tmp/.litevloc_integration_commit"
  ```

---

## Phase C — OpenNavMap Submodule and Cleanup

### Task 9: Add and pin LiteVLoc submodule

**Files:** `$OPENNAVMAP_ROOT/.gitmodules`, `$OPENNAVMAP_ROOT/third_party/litevloc_code`

- [ ] **Step 1: Retrieve integration commit hash if needed**

  ```bash
  # If $LITEVLOC_INTEGRATION_COMMIT is not set in current shell:
  export LITEVLOC_INTEGRATION_COMMIT=$(cat "$OPENNAVMAP_ROOT/tmp/.litevloc_integration_commit")
  echo "$LITEVLOC_INTEGRATION_COMMIT"
  ```

- [ ] **Step 2: Add submodule**

  ```bash
  cd "$OPENNAVMAP_ROOT"
  git submodule add -b opennavmap-integration \
    git@github.com:RPL-CS-UCL/litevloc_code.git \
    third_party/litevloc_code
  ```

- [ ] **Step 3: Pin submodule to specific commit**

  ```bash
  cd "$OPENNAVMAP_ROOT/third_party/litevloc_code"
  git checkout "$LITEVLOC_INTEGRATION_COMMIT"
  cd "$OPENNAVMAP_ROOT"
  git add third_party/litevloc_code .gitmodules
  git commit -m "chore: add LiteVLoc submodule pinned to opennavmap-integration"
  ```

- [ ] **Step 4: Verify submodule is pinned**

  ```bash
  git submodule status
  ```
  Expected: the commit hash shown matches `$LITEVLOC_INTEGRATION_COMMIT` (no leading `-` or `+`).

---

### Task 10: Update OpenNavMap docs and identity

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify: `docs/repo_structure_brief.md`

- [ ] **Step 1: Update README.md intro**

  Replace the project intro with:
  ```
  # OpenNavMap

  OpenNavMap is the main system for multi-session topometric map construction,
  submap merging, and scalable image-goal navigation experiments.

  LiteVLoc (https://github.com/RPL-CS-UCL/litevloc_code) is the visual localization
  submodule, included under third_party/litevloc_code.
  ```
  Also update any badge, installation, or usage section that still refers to the old `litevloc_private` name.

- [ ] **Step 2: Update CLAUDE.md identity section**

  At the top of CLAUDE.md, ensure the repo description reads:
  ```
  This repository is **OpenNavMap**: a multi-session topometric mapping and
  scalable image-goal navigation system.
  LiteVLoc (third_party/litevloc_code) is the visual localization submodule.
  ```

- [ ] **Step 3: Update docs/repo_structure_brief.md**

  Update the top-level structure block to reflect the new boundary:
  ```
  opennavmap/
  ├── python/
  │   ├── map_merge_pipeline.py
  │   ├── map_manager.py
  │   ├── benchmark_mms/
  │   ├── benchmark_vpr/
  │   ├── benchmark_kf_selection/
  │   └── utils/
  ├── scripts/
  ├── docs/
  ├── paper_writing/
  └── third_party/
      └── litevloc_code/   <- git submodule (RPL-CS-UCL/litevloc_code)
  ```

- [ ] **Step 4: Commit**

  ```bash
  cd "$OPENNAVMAP_ROOT"
  git add README.md CLAUDE.md docs/repo_structure_brief.md
  git commit -m "docs: rename repository identity to OpenNavMap"
  ```

---

### Task 11: Generate and run OpenNavMap core validation script

**Files:** `$OPENNAVMAP_ROOT/tmp/opennavmap_split_validation/validate_opennavmap_core_imports.py` (not committed)

- [ ] **Step 1: Ensure tmp/ is gitignored**

  ```bash
  grep -q "^tmp/" "$OPENNAVMAP_ROOT/.gitignore" || echo "tmp/" >> "$OPENNAVMAP_ROOT/.gitignore"
  git add .gitignore
  git commit -m "chore: ensure tmp/ is gitignored"
  ```

- [ ] **Step 2: Generate validation script**

  ```bash
  mkdir -p "$OPENNAVMAP_ROOT/tmp/opennavmap_split_validation"
  cat > "$OPENNAVMAP_ROOT/tmp/opennavmap_split_validation/validate_opennavmap_core_imports.py" << 'EOF'
  import os, pathlib, importlib, sys

  opennavmap_root = pathlib.Path(os.environ["OPENNAVMAP_ROOT"]).resolve()

  core_modules = ["map_merge_pipeline", "map_manager"]
  benchmark_spot = ["benchmark_vpr.evaluation", "benchmark_kf_selection.keyframe_selection"]
  shared_utils = ["utils.gtsam_pose_graph", "utils.utils_geom", "utils.utils_image"]

  failed = []
  for name in core_modules + benchmark_spot + shared_utils:
      try:
          mod = importlib.import_module(name)
          mod_file = pathlib.Path(mod.__file__).resolve()
          if opennavmap_root not in mod_file.parents:
              failed.append(f"FAIL {name}: {mod_file} not under OpenNavMap root")
          elif "third_party" in str(mod_file):
              failed.append(f"FAIL {name}: {mod_file} resolved from third_party (must be local)")
          else:
              print(f"OK   {name}: {mod_file}")
      except Exception as e:
          failed.append(f"FAIL {name}: import error — {e}")

  if failed:
      print("\n--- FAILURES ---")
      for f in failed:
          print(f)
      sys.exit(1)

  print("\nAll OpenNavMap core imports resolved locally — no LiteVLoc dependency.")
  EOF
  ```

- [ ] **Step 3: Run validation — standalone mode (no LiteVLoc in PYTHONPATH)**

  ```bash
  cd "$OPENNAVMAP_ROOT"
  PYTHONPATH="$OPENNAVMAP_ROOT/python" \
  OPENNAVMAP_ROOT="$OPENNAVMAP_ROOT" \
  python tmp/opennavmap_split_validation/validate_opennavmap_core_imports.py
  ```
  Expected: all lines `OK ...`, exits 0.
  If any FAIL: the file is missing from OpenNavMap or is being resolved from `third_party/`. Fix before proceeding.

- [ ] **Step 4: Skip legacy pytest suite for this migration**

  Do not run or require `python/test/` in this task. Those legacy tests depend on unavailable external packages (`pycpptools`) and LiteVLoc-owned modules that will be removed from OpenNavMap. The migration acceptance gate is Step 3's isolated path validation script.

---

### Task 12: Batch 1 cleanup — remove LiteVLoc-owned runtime Python modules

**Files:** delete from `$OPENNAVMAP_ROOT/python/`

- [ ] **Step 1: Reference scan before deletion**

  ```bash
  rg "loc_pipeline|global_planner|pose_fusion|depth_registration|camera_keyframe_select|image_graph|point_graph|image_node|point_node" \
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
  ```
  If any retained file imports a module being deleted: stop, fix the import, then re-run scan.

- [ ] **Step 2: Delete runtime modules**

  ```bash
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
  ```

- [ ] **Step 3: Re-run OpenNavMap validation**

  ```bash
  PYTHONPATH="$OPENNAVMAP_ROOT/python" \
  OPENNAVMAP_ROOT="$OPENNAVMAP_ROOT" \
  python tmp/opennavmap_split_validation/validate_opennavmap_core_imports.py
  ```
  Expected: still all `OK`. If any FAIL: a retained file had a hidden dependency — restore the deleted file (`git checkout -- <file>`), fix the dependency, then delete again.

- [ ] **Step 4: Commit batch 1**

  ```bash
  cd "$OPENNAVMAP_ROOT"
  git add -A
  git commit -m "cleanup: remove LiteVLoc-owned runtime modules (batch 1)"
  ```

---

### Task 13: Batch 2 cleanup — remove LiteVLoc-owned benchmark dirs

**Files:** delete `benchmark_map_free/` and `benchmark_rpe/` from `$OPENNAVMAP_ROOT/python/`

- [ ] **Step 1: Reference scan before deletion**

  ```bash
  rg "benchmark_map_free|benchmark_rpe" \
    "$OPENNAVMAP_ROOT/python/map_merge_pipeline.py" \
    "$OPENNAVMAP_ROOT/python/map_manager.py" \
    "$OPENNAVMAP_ROOT/python/benchmark_mms" \
    "$OPENNAVMAP_ROOT/python/benchmark_vpr" \
    "$OPENNAVMAP_ROOT/python/benchmark_kf_selection" || true
  ```
  If matches found: fix before deleting.

- [ ] **Step 2: Delete benchmark dirs**

  ```bash
  rm -rf "$OPENNAVMAP_ROOT/python/benchmark_map_free"
  rm -rf "$OPENNAVMAP_ROOT/python/benchmark_rpe"
  ```

- [ ] **Step 3: Re-run OpenNavMap validation**

  ```bash
  PYTHONPATH="$OPENNAVMAP_ROOT/python" \
  OPENNAVMAP_ROOT="$OPENNAVMAP_ROOT" \
  python tmp/opennavmap_split_validation/validate_opennavmap_core_imports.py
  ```
  Expected: all `OK`.

- [ ] **Step 4: Commit batch 2**

  ```bash
  cd "$OPENNAVMAP_ROOT"
  git add -A
  git commit -m "cleanup: remove LiteVLoc-owned benchmark dirs (batch 2)"
  ```

---

### Task 14: Batch 3 cleanup — remove LiteVLoc-only utils

**Files:** delete specific files from `$OPENNAVMAP_ROOT/python/utils/`

- [ ] **Step 1: Reference scan before deletion**

  ```bash
  rg "utils_vpr_method|utils_image_matching_method|pose_solver|utils_pipeline|utils_shortest_path|utils_ros|utils/benchmark" \
    "$OPENNAVMAP_ROOT/python/map_merge_pipeline.py" \
    "$OPENNAVMAP_ROOT/python/map_manager.py" \
    "$OPENNAVMAP_ROOT/python/benchmark_mms" \
    "$OPENNAVMAP_ROOT/python/benchmark_vpr" \
    "$OPENNAVMAP_ROOT/python/benchmark_kf_selection" || true
  ```
  If matches found: fix before deleting.

- [ ] **Step 2: Delete LiteVLoc-only utils**

  ```bash
  rm -f "$OPENNAVMAP_ROOT/python/utils/utils_vpr_method.py"
  rm -f "$OPENNAVMAP_ROOT/python/utils/utils_image_matching_method.py"
  rm -f "$OPENNAVMAP_ROOT/python/utils/pose_solver.py"
  rm -f "$OPENNAVMAP_ROOT/python/utils/pose_solver_default.py"
  rm -f "$OPENNAVMAP_ROOT/python/utils/utils_pipeline.py"
  rm -f "$OPENNAVMAP_ROOT/python/utils/utils_shortest_path.py"
  rm -rf "$OPENNAVMAP_ROOT/python/utils/utils_ros"
  rm -rf "$OPENNAVMAP_ROOT/python/utils/benchmark"
  rm -rf "$OPENNAVMAP_ROOT/python/test"
  ```
  **Do NOT remove:** `gtsam_pose_graph.py`, `utils_geom.py`, `utils_image.py` (shared, spec §6)
  **Do NOT remove:** `utils_map_merging.py`, `gen_covis_trav_edges.py` (OpenNavMap-owned)

- [ ] **Step 3: Re-run OpenNavMap validation**

  ```bash
  PYTHONPATH="$OPENNAVMAP_ROOT/python" \
  OPENNAVMAP_ROOT="$OPENNAVMAP_ROOT" \
  python tmp/opennavmap_split_validation/validate_opennavmap_core_imports.py
  ```
  Expected: all `OK`.

- [ ] **Step 4: Commit batch 3**

  ```bash
  cd "$OPENNAVMAP_ROOT"
  git add -A
  git commit -m "cleanup: remove LiteVLoc-only utils from OpenNavMap (batch 3)"
  ```

---

### Task 15: Batch 4a cleanup — remove LiteVLoc-owned scripts and launch files

**Files:** delete from `$OPENNAVMAP_ROOT/scripts/` and `$OPENNAVMAP_ROOT/launch/`

- [ ] **Step 1: Reference scan (check no retained Python file calls these)**

  ```bash
  rg "run_loc_pipeline|record_rosbag|export_odom_vloc|benchmark_mf|benchmark_rpe|run_finetune_rpe" \
    "$OPENNAVMAP_ROOT/python/map_merge_pipeline.py" \
    "$OPENNAVMAP_ROOT/python/map_manager.py" \
    "$OPENNAVMAP_ROOT/python/benchmark_mms" \
    "$OPENNAVMAP_ROOT/python/benchmark_vpr" \
    "$OPENNAVMAP_ROOT/python/benchmark_kf_selection" || true
  ```
  If matches found: fix before deleting.

- [ ] **Step 2: Delete LiteVLoc-owned scripts**

  ```bash
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
  ```

- [ ] **Step 3: Delete LiteVLoc-owned launch files**

  ```bash
  rm -f "$OPENNAVMAP_ROOT/launch/run_vloc_online_anymal.launch"
  rm -f "$OPENNAVMAP_ROOT/launch/run_vloc_online_simuenv.launch"
  rm -f "$OPENNAVMAP_ROOT/launch/run_vloc_offline_files.launch"
  rm -f "$OPENNAVMAP_ROOT/launch/run_pose_fusion.launch"
  rm -f "$OPENNAVMAP_ROOT/launch/run_navigation_interface_simuenv.launch"
  rm -f "$OPENNAVMAP_ROOT/launch/run_depth_reg.launch"
  ```

- [ ] **Step 4: Re-run OpenNavMap validation**

  ```bash
  PYTHONPATH="$OPENNAVMAP_ROOT/python" \
  OPENNAVMAP_ROOT="$OPENNAVMAP_ROOT" \
  python tmp/opennavmap_split_validation/validate_opennavmap_core_imports.py
  ```
  Expected: all `OK`.

- [ ] **Step 5: Commit batch 4a**

  ```bash
  cd "$OPENNAVMAP_ROOT"
  git add -A
  git commit -m "cleanup: remove LiteVLoc-owned scripts and launch files (batch 4a)"
  ```

---

### Task 16: Batch 4b cleanup — remove scripts confirmed in Task 3

**Files:** delete from `$OPENNAVMAP_ROOT/scripts/` only if confirmed LiteVLoc-owned in Task 3

- [ ] **Step 1: Delete scripts confirmed as LiteVLoc-owned in Task 3**

  Only execute the relevant lines based on Task 3 decisions:
  ```bash
  # Delete if confirmed LiteVLoc-owned in Task 3:
  rm -f "$OPENNAVMAP_ROOT/scripts/run_batch_gendata.sh"          # confirmed: LiteVLoc
  rm -f "$OPENNAVMAP_ROOT/scripts/run_batch_gendata_vpr.sh"      # confirmed: LiteVLoc
  rm -f "$OPENNAVMAP_ROOT/scripts/run_ego_blur.sh"              # confirmed: LiteVLoc
  rm -f "$OPENNAVMAP_ROOT/scripts/run_batch_extract_vpr_iqa.sh" # confirmed: LiteVLoc
  rm -f "$OPENNAVMAP_ROOT/scripts/run_batch_vpr_seq_slam.sh"    # confirmed: LiteVLoc
  ```
  If any of these were re-assigned to OpenNavMap in Task 3, skip the corresponding `rm`.

- [ ] **Step 2: Commit batch 4b**

  ```bash
  cd "$OPENNAVMAP_ROOT"
  git add -A
  git commit -m "cleanup: remove remaining LiteVLoc-owned scripts (batch 4b)"
  ```

---

### Task 17: Final push and verification

**Files:** git push, verify state

- [ ] **Step 1: Push all Phase C commits**

  ```bash
  cd "$OPENNAVMAP_ROOT"
  git push
  ```

- [ ] **Step 2: Final validation**

  ```bash
  PYTHONPATH="$OPENNAVMAP_ROOT/python" \
  OPENNAVMAP_ROOT="$OPENNAVMAP_ROOT" \
  python tmp/opennavmap_split_validation/validate_opennavmap_core_imports.py
  ```
  Expected: all `OK`.

- [ ] **Step 3: Check submodule is pinned**

  ```bash
  git submodule status
  ```
  Expected: commit hash matches `$LITEVLOC_INTEGRATION_COMMIT`, no leading `+` or `-`.

- [ ] **Step 4: Verify branch and remote**

  ```bash
  git branch -vv
  git remote -v
  git log --oneline -8
  ```
  Expected:
  - On `opennavmap_third_version`
  - Remote is `git@github.com:RPL-CS-UCL/opennavmap.git`
  - Recent commits reflect Phase C cleanup

- [ ] **Step 5: Check full checklist from spec §10**

  ```
  OpenNavMap:
  [ ] validate_opennavmap_core_imports.py passes
  [ ] map_merge_pipeline, map_manager __file__ under $OPENNAVMAP_ROOT/python
  [ ] gtsam_pose_graph, utils_geom, utils_image __file__ under $OPENNAVMAP_ROOT/python/utils (not third_party/)
  [ ] map_merge_pipeline.py --help succeeds
  [ ] benchmark_mms/, benchmark_vpr/, benchmark_kf_selection/ imports resolve
  [ ] third_party/litevloc_code submodule exists and is pinned
  [ ] README.md identifies repo as OpenNavMap
  [ ] CLAUDE.md identifies repo as OpenNavMap
  [ ] docs/repo_structure_brief.md reflects new boundary
  ```

---

## Notes

- **Do not rename `package.xml` or `CMakeLists.txt` project name** in this plan. The ROS package name change (`litevloc` → `opennavmap`) is out of scope for this split and can be done separately.
- **`segment_change/` and `ltl_task_planner/`** are not deleted in this plan. They are excluded from docs scope in Task 10. Physical deletion is a separate decision.
- **Temporary validation scripts** (`tmp/opennavmap_split_validation/`, `tmp/.litevloc_integration_commit`) are gitignored and not committed.
- **`third_party/vismatch/` and `third_party/VPR-methods-evaluation/`** are not moved in this plan — their placement follows whichever repo runs LiteVLoc's pipeline. This is a follow-up task.
