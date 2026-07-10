## Instraction of Running Map Merging

### Prepare the Submaps
The data is structured as follows:
```bash
data/0 # 0 is the submap id
  seq/000000.color.jpg       # color image 
      000001.color.jpg
      ...
  intrinsics.txt             # camera intrinsics (fx, fy, cx, cy, width, height)
  poses.txt                  # poses of each image represented as the map-free format (qw qx qy qz tx ty tz)
  poses_abs_gt.txt           # GT poses of each image represented as the map-free format (qw qx qy qz tx ty tz)
  timestamps.txt             # timestamps of each image
  edge_covis.txt             # edges between nodes to indicate covisibility relation (0: low visibility, 1: high visibility)
  edge_odom.txt              # edges between nodes to indicate odometry relation (connection means that nodes are closed)
  edge_trav.txt              # edges between nodes to indicate traversability relation (connection means that nodes are traversable)
  database_descriptors.txt   # 256-dimension CosPlace VPR global descriptors of each image
  gps_data.txt               # GPS data
  iqa_data.txt               # image quality score (0: bad quality due to low light, motion blur, etc, 100: high quality)
```


<!-- Format of out_general
  poses.txt: timestamp, tx, ty, tz, qw, qx, qy, qz
  intrinsics.txt: fx, fy, cx, cy, width, height
  gps.txt: timestamp, latitude, longitude, altitude, speed, accuracy


The format is same to the map-free benchmark

poses_abs.txt: absolute poses from the camera to the absolute world (e.g., groundtruth for evaluation)
```
# name qw qx qy qz tx ty tz
out_map0/seq/000000.color.jpg 4327.107000000 -0.605910000 1.012060000 1.000000000 -0.403850000 0.580440000 -0.580440000  0.403850
```

poses.txt: relative poses from the camera to the relative world (e.g., VIO trajectory), except for the reference submap
```
# name qw qx qy qz tx ty tz
out_map0/seq/000000.color.jpg 4327.107000000 -0.605910000 1.012060000 1.000000000 -0.403850000 0.580440000 -0.580440000  0.403850
``` -->