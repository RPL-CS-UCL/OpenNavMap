# OpenNavMap / LiteVLoc Repository Structure Overview

## 1. Repository Scope

This repository is the **OpenNavMap** system: a topometric mapping system for
**multi-session mapping** whose goal is to build, align, merge, and maintain a
lightweight topometric map — usable for navigation — from data captured over
multiple sessions.

**LiteVLoc** is the visual-localization submodule of this system, pinned as a git
submodule under `third_party/litevloc_code`. Its responsibilities are:

- Perform global visual localization against the final **multi-session topometric map**.
- Provide the visual anchoring from a goal image to a map node for image-goal navigation.
- Provide online visual pose estimates during navigation, working together with the
  planning/fusion modules.

The correct mental model of this repository is therefore a main/sub relationship:

- **OpenNavMap**: the main system — multi-session mapping, map management, map
  merging, graph-structure maintenance, and navigation support.
- **LiteVLoc**: the submodule — visual localization on top of an existing topometric map.

## 2. Code Lines

By responsibility, the repository splits into three lines:

1. **Multi-session map construction & merging** (main system objective)
2. **LiteVLoc visual localization** (`third_party/litevloc_code`)
3. **Navigation & system integration** (`third_party/litevloc_code`)

The first line is the system-level objective; the second builds on top of its output.

## 3. Top-Level Layout

```text
opennavmap/
├── python/            # OpenNavMap core mapping, merging, and map-level benchmarks
│   ├── map_merge_pipeline.py
│   ├── map_manager.py
│   ├── benchmark_mms/
│   ├── benchmark_vpr/
│   ├── benchmark_map_merge/
│   ├── benchmark_kf_selection/
│   └── utils/         # OpenNavMap-local utils: map merging / GTSAM / geom / image
├── launch/            # OpenNavMap launch entry points (LiteVLoc launches live in the submodule)
├── scripts/           # batch scripts and experiment entry points
├── docs/              # usage and workflow documentation
├── rviz_cfg/          # RViz configurations
├── third_party/       # in-repo third-party submodules / dependencies
│   └── litevloc_code/ # LiteVLoc visual-localization submodule
├── app/               # application-side code / interfaces
├── paper_writing/     # paper materials
├── requirements.txt   # Python dependencies
├── package.xml        # ROS package definition
└── CMakeLists.txt     # ROS/catkin build configuration
```

## 4. System Layers of `python/`

### 4.1 Multi-Session Map Construction & Merging

This part is the backbone of OpenNavMap.

- `python/map_merge_pipeline.py`
  - Main entry point for reading submaps, cross-graph matching, loop-closure
    establishment, GTSAM optimization, and map merging.
  - The core file for offline construction / alignment / fusion of the
    "multi-session topometric map".

- `python/map_manager.py`
  - Uniformly manages the multiple graph structures within one submap.
  - Currently manages: `odom`, `trav`, `covis`.

- `third_party/litevloc_code/python/image_graph.py` / `image_node.py`
  - Maintain the covisibility graph carrying images, descriptors, camera
    intrinsics, depth, etc.
  - Used as the map keyframe representation and as the map observation layer for LiteVLoc.

- `third_party/litevloc_code/python/point_graph.py` / `point_node.py`
  - Maintain the odometry graph and the traversability graph.
  - Oriented toward pose chains, traversability mapping, and shortest-path planning.

- `third_party/litevloc_code/python/utils/base_graph.py` / `base_node.py`
  - Base abstraction layer for graphs and nodes.

- `python/utils/gtsam_pose_graph.py`
  - GTSAM backend wrapper for map merging and pose-graph optimization.

- `python/utils/utils_geom.py` / `utils_image.py`
  - OpenNavMap-local shared utilities, kept so the map-merging core does not
    depend on the LiteVLoc submodule.

- `python/benchmark_mms/` / `benchmark_vpr/` / `benchmark_map_merge/` / `benchmark_kf_selection/`
  - OpenNavMap's map-level benchmarks and paper experiments.

At the system level, an OpenNavMap map is not a single structure but at least three
complementary graphs:

- **covis graph**: image keyframes and their visual associations
- **odom graph**: sequential pose chain
- **trav graph**: navigation reachability graph

Together these form the topometric map.

### 4.2 LiteVLoc Visual-Localization Submodule

This part builds on the map above.

- `third_party/litevloc_code/python/loc_pipeline.py`
  - LiteVLoc's core pipeline.
  - Reads the `covis graph` from the topometric map.
  - Runs "VPR coarse localization → image matching → pose solving".

- `third_party/litevloc_code/python/ros_loc_pipeline.py`
  - LiteVLoc online ROS wrapper.
  - Subscribes to image, depth, camera parameters, and fused pose; publishes `/vloc/odometry`.

- `third_party/litevloc_code/python/utils/utils_vpr_method.py`
  - VPR model initialization and retrieval / sequence-matching wrapper.

- `third_party/litevloc_code/python/utils/utils_image_matching_method.py`
  - Local image-matching wrapper.

- `third_party/litevloc_code/python/utils/pose_solver.py`
  - Pose solvers: PnP / Essential Matrix / Procrustes, etc.

LiteVLoc's job is not mapping, but rather:

- Use an **already-built multi-session map**.
- Perform global place recognition for a query image.
- Refine locally on candidate keyframes.
- Output the observing camera's pose in the map coordinate frame.

### 4.3 Navigation & System Integration

- `third_party/litevloc_code/python/global_planner.py`
  - Combines LiteVLoc's global matching with shortest-path planning over the `trav graph`.
  - Maps a goal image to a map node and generates waypoints.

- `third_party/litevloc_code/python/ros_global_planner.py`
  - Global-planning ROS wrapper.

- `third_party/litevloc_code/python/pose_fusion.py`
  - Fuses local odometry with visual localization via GTSAM.

- `third_party/litevloc_code/python/ros_pose_fusion.py`
  - Online pose-fusion ROS wrapper.

- `third_party/litevloc_code/python/depth_registration.py`
  - Additional local geometry / depth registration support.

This layer shows OpenNavMap is not just "map-file generation" — it extends into an
online navigation system.

## 5. `third_party/` and Other Components

The `third_party/` directory holds the external model components and the LiteVLoc
submodule that OpenNavMap depends on.

### 5.1 In-repo submodules

- `third_party/litevloc_code`
  - Source-of-truth for LiteVLoc visual localization, navigation runtime, pose
    fusion, and the map-free / RPE benchmarks.
  - OpenNavMap references it via a pinned commit.

- `third_party/vismatch`
  - Provides local image matching.
  - Used by `third_party/litevloc_code/python/utils/utils_image_matching_method.py`.
  - A key low-level component of LiteVLoc's local refinement stage.

### 5.2 External model components

- `third_party/VPR-methods-evaluation`
  - Supports VPR models, descriptor extraction, and the retrieval pipeline.
  - `utils_pipeline.py` and `utils_vpr_method.py` in the LiteVLoc submodule
    explicitly add it to `sys.path`.
  - It backs LiteVLoc's global visual-retrieval capability.

From a system-composition standpoint, OpenNavMap can be described as:

- **This repository**: OpenNavMap's core map construction, map merging, and
  map-level benchmark logic.
- **`third_party/litevloc_code`**: LiteVLoc visual localization, planning runtime,
  pose fusion, and localization-side benchmarks.
- **`third_party/vismatch`**: local image-matching submodule.
- **`third_party/VPR-methods-evaluation`**: external model module for global visual retrieval.

## 6. Recommended Reading Order

To quickly build a correct system-level understanding, read the code in this order:

1. `python/map_merge_pipeline.py`
2. `python/map_manager.py`
3. `third_party/litevloc_code/python/image_graph.py`
4. `third_party/litevloc_code/python/point_graph.py`
5. `third_party/litevloc_code/python/loc_pipeline.py`
6. `third_party/litevloc_code/python/global_planner.py`
7. `third_party/litevloc_code/python/pose_fusion.py`

This order follows OpenNavMap's real system hierarchy:

- First understand how the map is organized and merged.
- Then understand how LiteVLoc uses this map for visual localization.
- Finally understand how localization serves navigation and the online system.

## 7. One-Sentence Summary

**OpenNavMap is the main system repository whose goal is to build and maintain a
multi-session topometric map; LiteVLoc is the visual-localization submodule under
`third_party/litevloc_code` that performs global visual localization on that map
and further supports navigation and pose fusion.**
