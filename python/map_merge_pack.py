#!/usr/bin/env python
"""Tools for merged-map results that need no torch/gtsam: loop registry IO,
image index across step directories, consolidation of a step into a
self-contained map, and recovery of T_AB from a g2o file.

Usable as a library (navmap_console backend) and as a CLI:
    python python/map_merge_pack.py consolidate <step_dir> <out_dir> --images <dir> [--images <dir> ...]
    python python/map_merge_pack.py recover-registry <registry_txt> <g2o_path> <out_txt>
"""
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

RegistryRecord = Dict[str, object]
Registry = Dict[Tuple[int, int], RegistryRecord]

REGISTRY_HEADER = "# a_id,b_id,conf,first_step,reject_count,last_weight,tx,ty,tz,qx,qy,qz,qw"
MAP_FILES = [
	"timestamps.txt", "intrinsics.txt", "poses.txt", "poses_abs_gt.txt", "gps_data.txt",
	"iqa_data.txt", "edges_covis.txt", "edges_odom.txt", "edges_trav.txt", "database_descriptors.txt",
]
OPTIONAL_MAP_FILES = ["objects.json", "edges_object.txt"]


def pose_to_vec(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	"""4x4 -> (translation, quaternion xyzw)."""
	T = np.asarray(T, dtype=float)
	return T[:3, 3].copy(), Rotation.from_matrix(T[:3, :3]).as_quat()


def vec_to_pose(t: Sequence[float], q_xyzw: Sequence[float]) -> np.ndarray:
	T = np.eye(4)
	T[:3, :3] = Rotation.from_quat(np.asarray(q_xyzw, dtype=float)).as_matrix()
	T[:3, 3] = np.asarray(t, dtype=float)
	return T


def read_loop_registry(path: Path) -> Registry:
	"""Parse preds/loop_registry.txt. Legacy 6-column rows get T_AB=None."""
	registry: Registry = {}
	with open(path) as f:
		for lineno, line in enumerate(f, 1):
			line = line.strip()
			if not line or line.startswith("#"):
				continue
			parts = line.split(",")
			if len(parts) not in (6, 13):
				raise ValueError(f"{path}:{lineno}: expected 6 or 13 columns, got {len(parts)}")
			key = (int(parts[0]), int(parts[1]))
			record: RegistryRecord = {
				"conf": float(parts[2]),
				"first_step": int(parts[3]),
				"reject_count": int(parts[4]),
				"last_weight": float(parts[5]),
				"T_AB": None,
			}
			if len(parts) == 13:
				vals = [float(x) for x in parts[6:13]]
				record["T_AB"] = vec_to_pose(vals[:3], vals[3:])
			registry[key] = record
	return registry


def write_loop_registry(path: Path, registry: Registry) -> None:
	"""Write the 13-column format; every record must carry T_AB."""
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	lines = [REGISTRY_HEADER]
	for (a, b), rec in registry.items():
		if rec.get("T_AB") is None:
			raise ValueError(f"registry edge ({a},{b}) has no T_AB; run recover_registry_from_g2o first")
		t, q = pose_to_vec(rec["T_AB"])
		lines.append(
			f"{a},{b},{float(rec['conf']):.3f},{int(rec['first_step'])},{int(rec['reject_count'])},"
			f"{float(rec['last_weight']):.6f}," + ",".join(f"{v:.9f}" for v in (*t, *q))
		)
	path.write_text("\n".join(lines) + "\n")
