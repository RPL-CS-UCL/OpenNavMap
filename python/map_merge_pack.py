#!/usr/bin/env python
"""Tools for merged-map results that need no torch/gtsam: loop registry IO,
image index across step directories, consolidation of a step into a
self-contained map, and recovery of T_AB from a g2o file.

Usable as a library (navmap_console backend) and as a CLI:
    python python/map_merge_pack.py consolidate <step_dir> <out_dir> --images <dir> [--images <dir> ...]
    python python/map_merge_pack.py recover-registry <registry_txt> <g2o_path> <out_txt>
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


def _frame_id(name: str) -> Optional[int]:
	stem = Path(name).name
	if not stem.endswith(".color.jpg"):
		return None
	digits = stem[: -len(".color.jpg")]
	return int(digits) if digits.isdigit() else None


def build_image_index(source_dirs: Sequence[Path]) -> Dict[int, Path]:
	"""node id -> image path; later directories override earlier ones."""
	index: Dict[int, Path] = {}
	for d in source_dirs:
		seq = Path(d) / "seq"
		if not seq.is_dir():
			continue
		for img in seq.glob("*.color.jpg"):
			nid = _frame_id(img.name)
			if nid is not None:
				index[nid] = img
	return index


def recover_registry_from_g2o(registry: Registry, g2o_path: Path) -> Registry:
	"""Fill T_AB for legacy registry rows from the EDGE_SE3:QUAT lines of the same step's g2o."""
	edges: Dict[Tuple[int, int], np.ndarray] = {}
	with open(g2o_path) as f:
		for line in f:
			parts = line.split()
			if len(parts) >= 10 and parts[0] == "EDGE_SE3:QUAT":
				a, b = int(parts[1]), int(parts[2])
				vals = [float(x) for x in parts[3:10]]
				edges[(a, b)] = vec_to_pose(vals[:3], vals[3:])
	out: Registry = {}
	missing: List[Tuple[int, int]] = []
	for key, rec in registry.items():
		new_rec = dict(rec)
		if new_rec.get("T_AB") is None:
			if key in edges:
				new_rec["T_AB"] = edges[key]
			else:
				missing.append(key)
		out[key] = new_rec
	if missing:
		raise ValueError(f"{g2o_path}: no EDGE_SE3:QUAT for registry keys {missing[:10]}"
						 + (f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""))
	return out


def _link_or_copy(src: Path, dst: Path) -> None:
	"""Hard link (zero extra space); fall back to a copy across filesystems."""
	if dst.exists():
		dst.unlink()
	try:
		os.link(str(src), str(dst))
	except OSError:
		shutil.copy2(str(src), str(dst))


def _keys_of(path: Path) -> List[str]:
	with open(path) as f:
		return [line.split()[0] for line in f if line.strip()]


def consolidate_map(step_dir: Path, image_source_dirs: Sequence[Path], out_dir: Path,
					meta: Optional[Dict[str, object]] = None) -> Dict[str, object]:
	"""Turn one merge_* step directory into a self-contained, reloadable map.

	Copies the map text files (not preds/ except kf_vis/, not submap_disc_*), gathers every covis
	node's image from the lineage directories and writes a 13-column loop registry.
	"""
	step_dir, out_dir = Path(step_dir), Path(out_dir)
	pose_keys = _keys_of(step_dir / "poses.txt")
	bad = [f"line {i}: {k}" for i, k in enumerate(pose_keys) if k != f"seq/{i:06d}.color.jpg"]
	if bad:
		raise ValueError(f"{step_dir}/poses.txt frame names are not consecutive: " + "; ".join(bad[:5]))

	index = build_image_index(image_source_dirs)
	covis_keys = _keys_of(step_dir / "intrinsics.txt")
	missing = [k for k in covis_keys if _frame_id(k) not in index]
	if missing:
		raise FileNotFoundError(f"no image found for {len(missing)} covis node(s): "
								+ ", ".join(str(_frame_id(k)) for k in missing[:10]))

	(out_dir / "seq").mkdir(parents=True, exist_ok=True)
	(out_dir / "preds").mkdir(parents=True, exist_ok=True)
	kf_vis_src = step_dir / "preds" / "kf_vis"
	if kf_vis_src.is_dir():
		shutil.copytree(kf_vis_src, out_dir / "preds" / "kf_vis", dirs_exist_ok=True)
	else:
		(out_dir / "preds" / "kf_vis").mkdir(parents=True, exist_ok=True)
	for name in MAP_FILES + OPTIONAL_MAP_FILES:
		src = step_dir / name
		if src.is_file():
			shutil.copy2(str(src), str(out_dir / name))
	for k in covis_keys:
		_link_or_copy(index[_frame_id(k)], out_dir / k)

	registry_src = step_dir / "preds" / "loop_registry.txt"
	registry: Registry = read_loop_registry(registry_src) if registry_src.is_file() else {}
	if any(rec.get("T_AB") is None for rec in registry.values()):
		registry = recover_registry_from_g2o(registry, step_dir / "preds" / "initial_pose_graph.g2o")
	write_loop_registry(out_dir / "preds" / "loop_registry.txt", registry)

	result: Dict[str, object] = {
		"source_step_dir": str(step_dir),
		"num_nodes": len(pose_keys),
		"num_covis_nodes": len(covis_keys),
		"num_images": len(covis_keys),
		"registry_edges": len(registry),
		"created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
	}
	result.update(meta or {})
	with open(out_dir / "merge_meta.json", "w") as f:
		json.dump(result, f, indent=2, sort_keys=True)
	return result


def git_commit_of(repo: Path) -> Optional[str]:
	try:
		return subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
							  check=True, capture_output=True, text=True).stdout.strip()
	except (OSError, subprocess.CalledProcessError):
		return None


def main(argv: Optional[List[str]] = None) -> int:
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	sub = parser.add_subparsers(dest="cmd", required=True)
	c = sub.add_parser("consolidate", help="make a merge_* step self-contained (images hard-linked)")
	c.add_argument("step_dir", type=Path)
	c.add_argument("out_dir", type=Path)
	c.add_argument("--images", type=Path, action="append", default=[],
				   help="directories holding seq/*.color.jpg, in lineage order (later wins); step_dir is always last")
	r = sub.add_parser("recover-registry", help="add T_AB columns to a legacy 6-column loop_registry.txt")
	r.add_argument("registry_txt", type=Path)
	r.add_argument("g2o_path", type=Path)
	r.add_argument("out_txt", type=Path)
	args = parser.parse_args(argv)
	if args.cmd == "consolidate":
		meta = consolidate_map(args.step_dir, list(args.images) + [args.step_dir], args.out_dir)
		print(json.dumps(meta, indent=2))
	elif args.cmd == "recover-registry":
		fixed = recover_registry_from_g2o(read_loop_registry(args.registry_txt), args.g2o_path)
		write_loop_registry(args.out_txt, fixed)
		print(f"wrote {len(fixed)} edges to {args.out_txt}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
