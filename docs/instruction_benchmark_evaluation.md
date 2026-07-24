# Benchmark Evaluation

This document describes the released evaluation datasets, how each one maps to an
experiment in the paper, and how to run the corresponding benchmark. All released
imagery has human faces automatically anonymized (blurred with
[BlurryFaces](https://github.com/asmaamirkhan/BlurryFaces); see
`scripts/blur_seq_blurryfaces.py`). Only raw benchmark inputs are released —
results (`*_results_*`, `*_sfm_*`, `scene_stat`, `.rrd` visualizations) are excluded.

## 1. Dataset ↔ Experiment Correspondence

The paper (`OpenNavMap: Multi-Session Appearance-Based Topometric Mapping for
Scalable Visual Navigation`) benchmarks the localization and mapping stack on a
19 km self-collected dataset across four real-world environments. The three
released datasets correspond to the following experiments:

| Released dataset | Paper experiment | Task | Key metric |
|------------------|------------------|------|------------|
| `vpr_eval` | Exp. 1 — **Topological Localization** | Place retrieval / loop closure over a reference map | Precision@1, Recall@1 within `[7.5 m, 75°]` (Tab. II) |
| `map_free_eval` | Exp. 1 — **Metric Localization** | 6-DoF relative pose of a query image w.r.t. reference images (Map-Free format) | Precision@`[100 cm, 10°]`, AUC, translation error (Fig. 11) |
| `map_multisession_eval` | Exp. 2 & 3 — **Multi-Session Map Merging** | Merge independently captured submaps into one globally consistent map | ATE (RMSE): translational `[m]`, rotational `[deg]` (Tab. V–VII) |

## 2. Download

**🔗 [Download from Google Drive](https://drive.google.com/drive/folders/1Tpl3Leu0uo1b4iolLFpdfI5LO8CYCRe-)**

The release mirrors the expected local layout under `data_opennavmap/`. Each item
is packed as its own `.7z` archive. For **map merging** (`map_multisession_eval`),
every data folder is packed separately so you can download only what you need
(`s00000_orders.txt` travels with the `s00000_aria_data_390` archive of the same site).

| Archive | Content | Size |
|---------|---------|------|
| `map_free_eval/ucl_campus_aria.7z` | Metric localization — UCL Campus (Aria) | ~60 MB |
| `map_free_eval/360loc_aria.7z` | Metric localization — 360Loc (Aria) | ~124 MB |
| `vpr_eval/ucl_campus.7z` | Topological localization query/database — UCL Campus | ~250 MB |
| `map_multisession_eval/ucl_campus_aria/s00000_aria_data_000.7z` | Map merging — UCL Campus, session 000 | ~11 GB |
| `map_multisession_eval/ucl_campus_aria/s00000_aria_data_390.7z` | Map merging — UCL Campus, session 390 (+ `s00000_orders.txt`) | ~330 MB |
| `map_multisession_eval/ucl_campus_aria/s00003_exp_culling_aria_data_000.7z` | Map merging — UCL Campus, culling exp. 000 | ~2.4 GB |
| `map_multisession_eval/ucl_campus_aria/s00003_exp_culling_aria_data_390.7z` | Map merging — UCL Campus, culling exp. 390 (**small subset / quick start**) | ~73 MB |
| `map_multisession_eval/hkust_campus/s00000_aria_data_000.7z` | Map merging — HKUST Campus, session 000 | ~2.1 GB |
| `map_multisession_eval/hkust_campus/s00000_aria_data_390.7z` | Map merging — HKUST Campus, session 390 (+ `s00000_orders.txt`) | ~46 MB |
| `map_multisession_eval/hkust_campus/s00001_fp_data.7z` | Map merging — HKUST Campus, false-positive data | ~82 MB |
| `map_multisession_eval/vineyard/s00000_aria_data_000.7z` | Map merging — Vineyard, session 000 | ~440 MB |
| `map_multisession_eval/vineyard/s00000_aria_data_390.7z` | Map merging — Vineyard, session 390 (+ `s00000_orders.txt`) | ~16 MB |

### Download & Extract

```bash
# 1. Install the downloader and 7-Zip
pip install gdown
sudo apt install -y p7zip-full

# 2. Download the whole release folder (preserves the directory layout)
gdown --folder https://drive.google.com/drive/folders/1Tpl3Leu0uo1b4iolLFpdfI5LO8CYCRe-

# 3. Extract every archive in place (each archive already carries its sub-path)
cd data_release
for f in map_free_eval/*.7z vpr_eval/*.7z map_multisession_eval/*/*.7z; do
    7z x "$f" -o"$(dirname "$f")"
done
```

After extraction the data mirror the layout expected by the pipelines:

```
data_release/
├── map_free_eval/           # metric localization: ucl_campus_aria/, 360loc_aria/
├── vpr_eval/                # topological localization: ucl_campus/
└── map_multisession_eval/   # map merging: ucl_campus_aria/, hkust_campus/, vineyard/
```

## 3. Test-Time (Temporal & Spatial Coverage)

The self-collected data span multiple sites and long time horizons, which is what
makes multi-session localization and merging challenging (sparse spatial overlap,
temporal appearance shift, cross-device variance):

| Region / Scene | Distance | Time span | Sessions |
|----------------|----------|-----------|----------|
| R0 (UCL, in-order) | 0.6 km | 6 min | 4 |
| R1 (UCL, in-order) | 2.5 km | 18 hours | 7 |
| R2 (UCL, in-order) | 15.7 km | 110 days | 55 |
| 360Loc — Atrium / Concourse / Hall / Piatrium | 65×36 – 105×52 m | — | 4–5 |

`s00000_aria_data_390` and `s00000_aria_data_000` are the two per-site session
sets; `s00003_exp_culling_*` provides the culling/lifelong-maintenance subset;
`s00001_fp_data` holds false-positive stress cases; `s00000_orders.txt` encodes
the session ordering used by the **InOrder** and **Shuffled** merging protocols.

## 4. Running the Benchmarks

Set up `PYTHONPATH` first (both paths are required):

```bash
conda activate opennavmap
export PYTHONPATH=$(pwd)/python:$(pwd)/third_party/litevloc_code/python
```

### 4.1 Topological Localization — `vpr_eval`

Multi-stage place retrieval: VPR global retrieval → sequence matching
(`SingleMatch` / `SeqSLAM` / our DP matcher) → geometric verification (GV). Four
SoTA VPR models are supported (AnyLoc, NetVLAD, CosPlace, EigenPlaces). Correct
retrievals are those within `[7.5 m, 75°]` of a query; scored by Precision@1 and
Recall@1 (paper Tab. II).

```bash
# VPR / topological-localization benchmark
python python/benchmark_vpr/... --data_path <path>/vpr_eval/ucl_campus/<sXXXXX>
```

Each `sXXXXX` holds `query/` and `database/` sub-sequences (`out_map_*`), each with
`seq/`, `poses.txt`, `intrinsics.txt`, `timestamps.txt`.

### 4.2 Metric Localization — `map_free_eval`

6-DoF relative pose of a query image against a set of reference images, in the
[Map-Free](https://github.com/nianticlabs/map-free-reloc) format. Reported as
Precision@`[100 cm, 10°]` and AUC as a function of the number of reference images
`N`, plus translation error (paper Fig. 11–12, Tab. III–IV). Baselines include
HLoc (DISK+LG / SP+LG), VPR (CosPlace/NetVLAD), Reloc3R, DUSt3R, and MASt3R; our
method uses a GFM (DUSt3R/MASt3R) with confidence-map calibration.

```bash
# LiteVLoc offline localization (metric stage)
PYTHONPATH=$(pwd)/third_party/litevloc_code/python \
python third_party/litevloc_code/python/loc_pipeline.py \
    --map_path <path>/map_free_eval/<dataset>/map_free_eval/test/<sXXXXX> \
    --query_data_path <path>/map_free_eval/<dataset>/map_free_eval/test/<sXXXXX> \
    --image_size 512 288 --device cuda \
    --vpr_method cosplace --vpr_backbone ResNet18 --vpr_descriptors_dimension 256 \
    --img_matcher master --pose_solver pnp
```

Each `test/sXXXXX` contains `seq0/` (reference, identity pose) and `seq1/` (queries,
poses relative to the reference) plus `poses.txt`, `intrinsics.txt`. See the
LiteVLoc Map-Free benchmark guide:
[instruction_map_free_benchmark.md](https://github.com/RPL-CS-UCL/litevloc_code/blob/main/docs/instruction_map_free_benchmark.md).

### 4.3 Multi-Session Map Merging — `map_multisession_eval`

Incrementally align submaps into one global frame through four sequential stages —
topological localization → metric localization → PGO → node culling. Evaluated by
ATE (RMSE) against the Aria-SLAM ground truth, under two ordering protocols
(**InOrder**, **Shuffled**) and three inlier-threshold groups (paper Tab. V–VII).

```bash
# End-to-end map merging on one site
bash scripts/run_map_merging.sh          # or python python/map_merge_pipeline.py --help

# Map-merge benchmark (build submap SfM, then merge + evaluate)
bash python/benchmark_map_merge/scripts/run_baseline.sh --mode sfm
bash python/benchmark_map_merge/scripts/run_baseline.sh --mode merge
bash python/benchmark_map_merge/scripts/run_evaluation.sh --config map_merge.yaml
```

Trajectory evaluation uses `third_party/slam_trajectory_evaluation` (TUM format).
The **small subset** (`s00003_exp_culling_aria_data_390.7z`, ~73 MB) is enough to
run and inspect the pipeline without the full download. See the merging tutorial:
[instruction_map_merging.md](instruction_map_merging.md).
