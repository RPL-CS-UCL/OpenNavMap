## Submap Merge Pipeline

总体两阶段
--submap-sfm   → 每个 submap 独立建 SfM → submaps_sfm/{sid}/sfm/
--submap-merge → 增量合并所有 SfM       → results_.../merge_.../

### 阶段一：build_submap_sfm()（每个 submap）
1. 读 poses.txt（VIO W2C），按 sfm_sample_dist=0.25m 空间降采样得 sfm_images
2. SuperPoint 提取特征（全帧）+ NetVLAD 全局特征
3. VIO 引导配对 → LightGlue 匹配
4. _build_vio_reference_model()：用 VIO poses 初始化 COLMAP 模型
5. _triangulate_with_vio_prior()：三角化 3D 点（可选 BA，pose 固定）
6. 写 submaps_sfm/{sid}/sfm/（COLMAP 二进制）

### 阶段二：增量合并（merge_model_with_se3()）
Sub0 初始化：直接加载预建 SfM 作为 model0（W0 坐标系），用 reindex_dict 全局重命名帧号，写 submap_disc_0/poses.txt
Sub1–SubN 逐个合并：
sub_i SfM（Wi 坐标系）
    ↓
NetVLAD 检索：inc 帧 → model0 所有注册帧，top-10 候选
    ↓
SuperPoint+LightGlue 匹配 + 几何验证（Fundamental 内点 ≥100）
    ↓
_run_pnp()：每个采样帧在 W0 中定位，得 {frame: C2W in W0}
    ↓
_estimate_se3_umeyama()：RANSAC Umeyama 对齐相机中心，得 T(Wi→W0)
    ↓
刚性变换 model_i 所有 3D 点 + 所有帧 → add 进 model0
    ↓
更新 merged_poses，写 submap_disc_0/poses.txt + preds/merged_pred.txt

### TODO:
~~ "_run_pnp()：每个采样帧在 W0 中定位，得 {frame: C2W in W0}" -> 针对pnp的结果得 {frame: C2W in W0}，评估跟GT之间的error。方法是去查询该图像的poses_abs.txt（mapfree格式）的结果，计算该图像的GT位姿跟W0的GT位姿的相对平移，然后计算其error，然后输出到*merge_stats.json-pnp_per_frame中。先针对ucl_campus_aria submap0和submap1做实验，把结果输出到/Titan/dataset/data_opennavmap/map_multisession_eval/ucl_campus_aria/s00000_results_in_2sub_hloc_sfm_netvlad_splg_full_data_sba0~~

~~ ucl_campus_aria/s00000_sfm_netvlad_splg_data_sba0/submaps_sfm/{0...54}/sfm中每个submap中总共能够正确三角化的landmark数量，并输出到/Titan/code/robohike_ws/src/opennavmap/python/benchmark_map_merge/docs/landmark-count-sfm-netvlad_splg.md合适位置~~

~~ucl_campus_aria/s00000_sfm_netvlad_splg_full_data_sba0/submaps_sfm/{0...54}/sfm中每个submap中总共能够正确三角化的landmark数量，并输出到/Titan/code/robohike_ws/src/opennavmap/python/benchmark_map_merge/docs/landmark-count-sfm-netvlad_splg.md合适位置~~

~~1. 我看你计算的pnp error都是偏大，仔细检查你的计算pnp error是否正确，坐标系是否正确。~~
~~2. 根据poses_abs.TXT可以知道任意两个图像的相对平移和旋转。调整geometric verification的finliners的阈值为200，然后判断>200的图像pair中，计算其相对的GT translation 和 rotation是否 7m且 90度，如果符合则在*merge_stats.json标注TP，否则标注FP，并且统计TP和FP的数量和比例，并且统计pnp计算中er`ror>2m和<2m的数量和比例的。然后在重新拼接submap0和submap1~~

~~1. 解决序号编号的问题~~

Naming Rule:
    Dataset name:
        s00000_aria_data_{sample_distance} like s00000_aria_data_390 (sampling distance is 390cm), s00000_aria_data_025 (sampling distance is 25cm) like s00000_aria_data_000 {no sample distance, full dataset}

    Approach name:
        kf_spgo_cc_seqmatch
        kf_spgo_cc_seqmatch_master
        hloc_sfm_netvlad_splg: the approach using data same to kf_spgo_cc_seqmatch on the s00000_aria_data_390
        hloc_sfm_netvlad_splg_025: the approach using the frame with keyframe threshold s00000_aria_data_025 (different from above approaches), using the hloc-based implementation using netvlad+superpoint+lightglue
        hloc_sfm_netvlad_disklg_025: the approach using the frame with keyframe threshold s00000_aria_data_025 (different from above approaches), using the hloc-based implementation using netvlad+disk+lightglue
        hloc_sfm_netvlad_disklg_025: the approach using the frame with keyframe threshold s00000_aria_data_025 (different from above approaches), using the hloc-based implementation using netvlad+disk+lightglue

    Results_name:
        s00000_results_{order like "in" "r0" "r1" "r2" "r3" "r4" "r5" "r6" "r7" "r8"}_{approach_name}
        For example:
            s00000_results_in_hloc_sfm_netvlad_splg_025
        Immediate SFM results:
            s00000_sfm_netvlad_splg_025
            s00000_sfm_netvlad_splg_390

    Evaluation:
        Yaml: OpenNavMap_map_merge.yaml
        Script: run_traj_eval_OpenNavMap_map_merge.sh