Format of out_general
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
```