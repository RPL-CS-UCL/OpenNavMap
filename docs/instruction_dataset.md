# Dataset

## Download

The evaluation datasets are released on Google Drive (faces anonymized):

**🔗 [OpenNavMap data_release](https://drive.google.com/drive/folders/1Tpl3Leu0uo1b4iolLFpdfI5LO8CYCRe-)**

See the [Testing Data](../README.md#-testing-data) section in the README for the archive list and the download/extract commands.

## Dataset Processing

Pipeline overview:

```
raw data -> raw_data_out_general -> data for specific tasks
```

## Map Data Format

Each map/submap directory follows the layout below (mapfree convention):

```
map_root/
├── seq/                     # image frames
├── timestamps.txt           # img_name timestamp
├── intrinsics.txt           # frame_path fx fy cx cy width height
├── poses.txt                # frame_path qw qx qy qz tx ty tz (world-to-camera)
├── poses_abs_gt.txt         # optional, absolute pose GT
├── gps_data.txt             # optional
├── iqa_data.txt             # optional, image quality assessment
├── edges_covis.txt          # [node_a, node_b, weight]
├── edges_odom.txt
├── edges_trav.txt
└── database_descriptors.txt # VPR descriptors
```

- `poses.txt` uses the mapfree format: `R(q), t` transform a world point into the camera frame (`Rp + t`).
- `seq0/frame_00000.jpg` is always the identity pose; query poses are given relative to the reference frame.
