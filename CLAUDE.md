# CLAUDE.md

## Overview

**OpenNavMap**: a multi-session topometric mapping + image-goal navigation system.
**LiteVLoc** (`third_party/litevloc_code`) is a **required** submodule that must be
initialized; it provides visual localization, the graph structures (`image_graph.py`,
`point_graph.py`, etc.), and the shared `utils/` helper functions.

When running any OpenNavMap script, `PYTHONPATH` must include both paths:
```bash
export PYTHONPATH=$(pwd)/python:$(pwd)/third_party/litevloc_code/python
```

## Common Commands

```bash
# Set up the environment (Python 3.8 + CUDA 11.8)
conda create --name opennavmap python=3.8
conda activate opennavmap
conda install pytorch=2.0.1 torchvision=0.15.2 pytorch-cuda=11.8 numpy=1.24.3 -c pytorch -c nvidia
pip install -r requirements.txt

# Verify the torch installation
python test_torch_install.py

# Build the ROS package (optional)
catkin build opennavmap -DPYTHON_EXECUTABLE=$(which python)

# Sanity-check the OpenNavMap core import
PYTHONPATH=$(pwd)/python:$(pwd)/third_party/litevloc_code/python python python/map_merge_pipeline.py --help

# LiteVLoc offline localization pipeline
PYTHONPATH=$(pwd)/third_party/litevloc_code/python python third_party/litevloc_code/python/loc_pipeline.py \
    --map_path <map_dir> \
    --query_data_path <query_dir> \
    --image_size 512 288 --device=cuda \
    --vpr_method cosplace --vpr_backbone=ResNet18 --vpr_descriptors_dimension=256 \
    --img_matcher master \
    --pose_solver pnp --config_pose_solver third_party/litevloc_code/python/config/dataset/matterport3d.yaml

# ROS online localization (simulation / real robot)
roslaunch litevloc run_vloc_online_simuenv.launch
roslaunch litevloc run_vloc_online_anymal.launch

# Map merging
bash scripts/run_map_merging.sh
```

## Directory Structure

```
python/
├── map_merge_pipeline.py   # main entry for multi-session map construction & merging
├── map_manager.py          # multi-graph coordination/management
├── utils_map_merging.py    # map-merging utilities (OpenNavMap-specific)
├── gen_covis_trav_edges.py # covis/trav edge generation script (OpenNavMap-specific)
├── benchmark_mms/          # multi-session mapping benchmark
├── benchmark_vpr/          # VPR evaluation
├── benchmark_kf_selection/ # keyframe-selection evaluation
└── benchmark_map_merge/    # map-merging evaluation

third_party/litevloc_code/python/
├── loc_pipeline.py         # LiteVLoc offline localization entry
├── ros_loc_pipeline.py     # LiteVLoc online localization ROS wrapper
├── global_planner.py       # global planning over the trav graph
├── pose_fusion.py          # odometry + visual-localization fusion
├── image_graph.py          # ImageGraph structure (shared with OpenNavMap)
├── point_graph.py          # PointGraph structure (shared with OpenNavMap)
├── utils/                  # shared helpers (used by both OpenNavMap and LiteVLoc)
└── config/dataset/         # YACS dataset configs (single source of truth)
```

## Map Data Format

```
map_root/
├── seq/                        # image frames
├── timestamps.txt              # img_name timestamp
├── intrinsics.txt              # per-frame: frame_path fx fy cx cy width height
├── poses.txt                   # per-frame: frame_path qw qx qy qz tx ty tz
├── poses_abs_gt.txt            # optional, absolute pose GT
├── gps_data.txt                # optional
├── iqa_data.txt                # optional, image quality assessment
├── edges_covis.txt             # [node_a, node_b, weight]
├── edges_odom.txt
├── edges_trav.txt
└── database_descriptors.txt    # VPR descriptors
```

**poses.txt (mapfree format):** `frame_path qw qx qy qz tx ty tz`
- world-to-camera: `R(q), t` transform a world point into the camera frame, i.e. `Rp + t`.
- `seq0/frame_00000.jpg` is always the identity pose; query poses are given relative to the reference frame.

## Released Datasets ↔ Experiments

The evaluation datasets are released on Google Drive
([data_release folder](https://drive.google.com/drive/folders/1Tpl3Leu0uo1b4iolLFpdfI5LO8CYCRe-);
human faces anonymized). Each dataset maps to one paper experiment:

| Dataset | Paper experiment | Task | Key metric |
|---------|------------------|------|------------|
| `vpr_eval` | Exp 1 — Topological Localization | place retrieval / loop closure over a reference map | Precision@1, Recall@1 @ `[7.5 m, 75°]` |
| `map_free_eval` | Exp 1 — Metric Localization | 6-DoF query pose w.r.t. reference images (Map-Free format) | Precision@`[100 cm, 10°]`, AUC |
| `map_multisession_eval` | Exp 2 & 3 — Map Merging | merge multi-session submaps into a globally consistent map | ATE (trans `[m]` / rot `[deg]`, RMSE) |

- Only raw benchmark inputs are released; `*_results_*`, `*_sfm_*`, `scene_stat`, and `.rrd` files are excluded.
- See [docs/instruction_benchmark_evaluation.md](docs/instruction_benchmark_evaluation.md) for the download list, test-time coverage, and run commands.

## benchmark_map_merge

- **Directory naming convention:**
  - Data directory: `s00000_aria_data_000`
  - SfM result: `s00000_sfm_netvlad_splg_{dist}` (`dist = f"{int(sfm_sample_dist*100):03d}"`, e.g. `0.25` → `_025`)
  - Merge result: `s00000_results_{order_tag}_{method}_{dist}` (no `_sba{n}` suffix)
  - No suffix is appended when `dist=0`.

- Evaluation uses `third_party/slam_trajectory_evaluation` (not `evo`). After merging, `export_to_eval_structure()` is called automatically to write TUM trajectories to `/Titan/dataset/data_opennavmap/traj_eval_data/map_merge_eval_data`.

- Script entry points (`python/benchmark_map_merge/scripts/`):
  ```bash
  bash run_baseline.sh --mode sfm                                    # build SfM for all submaps
  bash run_baseline.sh --mode sfm --max-submaps 2 --overwrite        # only the first 2
  bash run_baseline.sh --mode merge --max-submaps 2 --overwrite      # merge and evaluate
  bash run_evaluation.sh --config map_merge.yaml                     # run evaluation standalone
  ```

- **Reference merge configuration.** `scripts/run_map_merging.sh` now defaults to the
  configuration validated end-to-end on the full 55-submap sequence, so the reference run needs
  no overrides:
  ```bash
  bash scripts/run_map_merging.sh s00000 0 <method_tag> master 1 1 1
  ```
  It resolves to `--pgo_robust gnc_gm`, `--pgo_loop_sigma_trans 0.1` / `--pgo_loop_sigma_rot 1.0`,
  `--pgo_loop_conf_scaling inverse`, with `REFINE_CONF_THRESHOLD = 0.5`
  (`python/utils_map_merging.py`). Persistent loop factors are unconditional: accepted
  inter-submap edges stay loop factors across merge steps so GNC can re-classify an earlier
  bad edge. Change one knob at a time and re-run the full sequence —
  the merge is incremental and path-dependent, so replaying only the final step does not
  predict what a different setting would have produced.

- **Comparing ATE across runs.** Different runs keep different node counts, and the merged
  `poses.txt` renumbers frames, so frame names are *not* comparable between runs. Join on the
  absolute GT pose in `poses_abs_gt.txt` and evaluate on the frame set the runs share;
  also report each run's node count and its number of `submap_disc_*` components, since a run
  whose final map split into several components is not directly comparable to one that did not.
  Merge runs are also not bit-reproducible (GPU non-determinism diverges the intermediate
  trajectory), so read trends and final values, not per-step spikes.

## Known Issues

- `cannot import name 'cache' from 'functools'`: replace with `functools.lru_cache(maxsize=None)`.
- `libffi/libtiff` symlink issue (ARM): manually rebuild the `.so` symlinks in the conda environment.
- `cannot allocate memory in static TLS block`: add `export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1` to the launch script.
- `libGL error: MESA-LOADER: failed to open iris/swrast` + `Could not create GL context` when opening a
  3D viewer (e.g. `show_reconstruction()` in `third_party/pose_estimation_models`). Two causes stack up:
  1. A shell-level `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGL.so:...libGLEW.so` forces the *system* libGL
     into a conda process that already loads its own, so Mesa falls back to conda's build-time DRI search
     path `/usr/lib/dri` — a directory that does not exist. The real drivers live in
     `/usr/lib/x86_64-linux-gnu/dri/`.
  2. System Mesa 21.2.6 does not know recent Intel iGPUs (`MESA: warning: Driver does not support the
     0xa780 PCI ID`), so hardware `iris` cannot start and software rendering is required.

  Run the viewer with the preload masked for that process only (keep `LD_PRELOAD` globally — other
  components need it):
  ```bash
  env -u LD_PRELOAD \
      LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri \
      LIBGL_ALWAYS_SOFTWARE=1 \
      python main_estimator.py --model vggt --scene_root <scene_dir>
  ```
  This renders through `llvmpipe` (CPU), which is slow on large point clouds. Upgrading system Mesa to
  >= 23.x (`ppa:kisak/kisak-mesa`) restores hardware acceleration; `LIBGL_ALWAYS_SOFTWARE` can then be
  dropped and only `LIBGL_DRIVERS_PATH` kept. Inference and pose estimation are unaffected either way.
- `Unable to show reconstruction: trimesh.viewer.windowed requires pip install "pyglet<2"`: trimesh's
  windowed viewer never adopted the pyglet 2.x API. Run `pip install "pyglet<2"` (pins 1.5.31). Nothing
  else in the environment depends on pyglet, so the downgrade is safe. Surfaces after the GL fix above,
  since the viewer only gets that far once a GL context can be created.
- Viewer runs block on the GUI event loop, so `print()` output stays in stdout's block buffer when the
  run is redirected to a file and later killed — the pose result looks missing even though it was
  computed. Use `PYTHONUNBUFFERED=1` when logging a viewer run to a file.
- **Random memory corruption in long `run_map_merging.sh` runs — CPU-level, not a software bug.**
  Three events on the same host, all inside `utils/vpr_dp.py::_perform_graph_search` (a ~39 s
  single-threaded pure-Python triple loop, reached right after the `D_all shape: (...)` log line):
  merge step 41 and step 46 died with `segfault at 0 ip 0000000000000000` (a jump through a NULL
  function pointer), and step 22 raised `TypeError: 'range_iterator' object is not callable` from
  `next_states.get(...)` where `next_states` is a plain `dict`. That last one is the tell — the
  iterator built by the `for k in range(...)` on the line above had displaced the local in the
  CPython frame. No software fault can produce that; it is a bit flip, and this host's memory is
  non-ECC (`EDAC ie31200: No ECC support`), so nothing detects or corrects it.

  Both segfaults were logged on **CPU 14**. On this i9-14900K, CPUs 12–15 are the 6.0 GHz "favored"
  cores (the other P-cores cap at 5.7 GHz, E-cores at 4.4 GHz), and Intel ITMT deliberately parks
  long single-threaded work on them — exactly this loop. Raptor Lake i9 parts are subject to the
  known Vmin-shift/oxidation degradation, which hits the highest-clocked cores first; microcode
  0x133 mitigates further degradation but cannot undo damage already done.

  `scripts/run_map_merging.sh` now pins the run away from those cores automatically
  (`detect_non_boost_cpus`, override via `MERGE_CPU_LIST`, empty to disable). This costs ~12% of CPU
  throughput. It matters because `map_merge_pipeline.py` has **no resume logic**: every crash costs
  a full re-run from step 0. If corruption persists after pinning, the degradation has spread beyond
  the favored cores and the fix is a BIOS-level offset (or an RMA — Intel extended the warranty on
  these parts to 5 years).

  Two earlier suspects were ruled out as *contributing* rather than causal, but their fixes are worth
  keeping: the script used to write `LD_PRELOAD="${LD_PRELOAD:-<conda>/lib/libstdc++.so.6}"`, and `:-`
  keeps an already-set value, so a shell-level `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGL.so:...`
  force-loaded the system GL stack ahead of Python (same trap as the GL issue above); the process also
  hosts three BLAS builds and two OpenMP runtimes. The script now pins `LD_PRELOAD` and sets
  `MKL_THREADING_LAYER=GNU`. Crashes continued after both, which is what pointed at hardware.

## third_party Dependencies

- `third_party/litevloc_code`: **must be initialized** (`git submodule update --init --recursive`). Without it the OpenNavMap main pipeline cannot run.
- `third_party/vismatch`: image-matching dependency, used by `litevloc_code/utils/utils_image_matching_method.py`.
- `third_party/VPR-methods-evaluation`: VPR retrieval dependency, used by `python/utils_map_merging.py`.
