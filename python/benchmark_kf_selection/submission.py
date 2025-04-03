#! /usr/bin/env python

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from zipfile import ZipFile
import time
import numpy as np
from tqdm import tqdm
from transforms3d.quaternions import mat2quat
from multiprocessing import Pool
from collections import defaultdict

# Custom module imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
from utils.utils_geom import read_intrinsics, read_poses, read_descriptors, read_img_names
from utils.utils_geom import convert_vec_to_matrix, convert_matrix_to_vec, correct_intrinsic_scale
from utils.utils_image import load_rgb_image, load_depth_image
from utils.utils_image_matching_method import save_visualization
from utils.utils_vpr_method import perform_knn_search
from utils.pose_solver import available_solvers, get_solver
from benchmark_rpe.rpe_default import cfg
from image_node import ImageNode
from image_graph import ImageGraph
from keyframe_selection import DB_Ratio

# Matching framework imports
from matching import available_models, get_matcher
from matching.utils import to_numpy

@dataclass
class PoseResult:
    image_name: str
    q: np.ndarray
    t: np.ndarray
    inliers: float

    def __str__(self) -> str:
        formatter = {"float": lambda v: f"{v:.6f}"}
        max_line_width = 1000
        q_str = np.array2string(
            self.q, formatter=formatter, max_line_width=max_line_width
        )[1:-1]
        t_str = np.array2string(
            self.t, formatter=formatter, max_line_width=max_line_width
        )[1:-1]
        return f"{self.image_name} {q_str} {t_str} {self.inliers}"
def predict(graph, queries, matcher, solver, scene, out_dir, args):
	results_dict = defaultdict(list)
	running_time = []
	db_descs = np.array([node.get_descriptor() for node in graph.nodes.values()], dtype=np.float32)

	for query_node in tqdm(queries, desc="Processing queries"):
		try:
			start_time = time.time()
			
			# Perform descriptor matching
			query_desc = query_node.get_descriptor().reshape(1, -1)
			_, pred = perform_knn_search(db_descs, query_desc, query_desc.shape[1], [1])
			map_node = list(graph.nodes.values())[pred[0][0]]

			# Image matching
			match_result = matcher(map_node.rgb_image, query_node.rgb_image)
			mkpts0, mkpts1 = match_result["inlier_kpts0"], match_result["inlier_kpts1"]
			matching_time = time.time() - start_time

			# Coordinate transformation
			mkpts0_raw = mkpts0 * [
				map_node.raw_img_size[0] / map_node.img_size[0],
				map_node.raw_img_size[1] / map_node.img_size[1]
			]
			mkpts1_raw = mkpts1 * [
				query_node.raw_img_size[0] / query_node.img_size[0],
				query_node.raw_img_size[1] / query_node.img_size[1]
			]

			# Pose estimation
			start_solve = time.time()
			depth_img = to_numpy(query_node.depth_image.squeeze(0))
			R, t, inliers = solver.estimate_pose(
				mkpts1_raw, mkpts0_raw,
				query_node.raw_K, map_node.raw_K,
				depth_img, None
			)
			solve_time = time.time() - start_solve
			
			# Store timing results
			running_time.append(matching_time + solve_time)

			# Convert to quaternion and store
			if np.isnan(R).any() or np.isnan(t).any():
				estimated_pose = PoseResult(
					image_name=query_node.id,
					q=mat2quat(np.eye(3)).reshape(-1),
					t=np.zeros(3),
					inliers=-1
				)
				results_dict[scene].append(estimated_pose)

				raise ValueError(f"Invalid pose estimation for {query_node.id}")

			T_mapnode_curr = np.eye(4)
			T_mapnode_curr[:3, :3], T_mapnode_curr[:3, 3] = R, t.reshape(3)
			T_mapnode = convert_vec_to_matrix(map_node.trans_gt, map_node.quat_gt, mode='xyzw')
			T_w_curr = T_mapnode @ T_mapnode_curr
			T_curr_w = np.linalg.inv(T_w_curr)
			R_inv, t_inv = T_curr_w[:3, :3], T_curr_w[:3, 3]

			estimated_pose = PoseResult(
				image_name=query_node.id,
				q=mat2quat(R_inv).reshape(-1),
				t=t_inv.reshape(-1),
				inliers=inliers
			)
			results_dict[scene].append(estimated_pose)

			# Visualization
			if args.debug:
				print(f"{args.keyframe_selector}-Match number: {inliers} between {map_node.id} and {query_node.id}")
				print(f"{args.keyframe_selector}-GT: {query_node.trans_gt}")
				print(f"{args.keyframe_selector}-EST: {T_w_curr[:3, 3].T}")

				Path(out_dir/"preds").mkdir(exist_ok=True, parents=True)
				save_visualization(
					query_node.rgb_image, map_node.rgb_image,
					mkpts0, mkpts1,
					out_dir,
					f"{map_node.id.replace('/', '_').replace('.jpg', '')}_" + \
					f"{query_node.id.replace('/', '_').replace('.jpg', '')}",
					n_viz=100
				)

		except Exception as e:
			tqdm.write(f"Error processing {query_node.id}: {str(e)}")
			continue

	return results_dict, np.mean(running_time)

def save_submission(results_dict, output_path):
	with ZipFile(output_path, "w") as zip_file:
		for scene, poses in results_dict.items():
			content = "\n".join(str(pose) for pose in poses)
			zip_file.writestr(f"pose_{scene}.txt", content.encode("utf-8"))

def eval(args):
	cfg.merge_from_file(args.config)

	output_root = Path(args.out_dir)
	output_root.mkdir(parents=True, exist_ok=True)
	for scene in cfg.DATASET.TEST_SCENES:
		keyframe_path = Path(args.keyframe_dir) / args.split / scene
		if not os.path.exists(keyframe_path):
			print(f"Keyframe path {keyframe_path} not exist")
			continue
		
		scene_path = Path(args.dataset_dir) / args.split / scene
		# print(f'Processing scene: {scene}')

		# Load base data
		poses, intrinsics, descs, keyframes, submap_splits = \
			load_data(scene_path, keyframe_path, args)
		
		# Split database and query sets
		num_query = max(1, int(len(submap_splits) * (1 - DB_Ratio)))
		num_database = len(submap_splits) - num_query
		db_submaps = submap_splits[:num_database]
		query_submaps = submap_splits[num_database:]

		# Build database graph
		graph = ImageGraph(scene_path)
		for submap in db_submaps:
			for img_name in submap['frames']:
				if img_name in keyframes:
					node = create_image_node(
						cfg,
						scene_path, img_name, args.image_size, 
						descs[img_name], intrinsics[img_name], poses[img_name]
					)
					graph.add_node(node)
		# print(f'Map with {graph.get_num_node()} Nodes')
		# print(', '.join([node.id for node in graph.nodes.values()]))

		# Prepare query set
		queries = []
		for submap in query_submaps:
			for img_name in submap['frames']:
				queries.append(create_image_node(
					cfg,
					scene_path, img_name, args.image_size,
					descs[img_name], intrinsics[img_name], poses[img_name]
				))

		# Process with different matchers	
		with open(output_root / "runtimes.txt", "w") as timing_file:
			for model in args.image_match_models:
				matcher = get_matcher(model, args.device)
				solver = get_solver(args.pose_solver, cfg)
				
				model_dir = output_root / f"{model}_{args.pose_solver}_{args.keyframe_selector}"
				model_dir.mkdir(exist_ok=True)
				results, avg_time = predict(graph, queries, matcher, solver, scene, model_dir, args)
				
				# Save results
				save_submission(results, model_dir/"submission.zip")
				
				# Record timing
				timing_str = f"{model}_{args.pose_solver}_{args.keyframe_selector}: {avg_time:.2f}s"
				timing_file.write(timing_str + "\n")
				tqdm.write(timing_str)

def load_data(scene_path, keyframe_path, args):
	return (
		read_poses(Path(scene_path)/"poses.txt"),
		read_intrinsics(Path(scene_path)/"intrinsics.txt"),
		read_descriptors(Path(keyframe_path)/"descriptors.txt"),
		read_img_names(Path(keyframe_path)/f"keyframes_{args.keyframe_selector}.txt"),
		np.load(Path(keyframe_path)/"submap_split.npy", allow_pickle=True)
	)

def create_image_node(cfg, scene_path, img_name, resize, desc, intr, pose):
	rgb_img = load_rgb_image(scene_path/ img_name, resize)
	depth_img = load_depth_image(scene_path/ img_name.replace('jpg', f'{cfg.DATASET.ESTIMATED_DEPTH}.png'))
	
	raw_K = np.array([intr[0], 0, intr[2], 0, intr[1], intr[3], 0, 0, 1]).reshape(3,3)
	raw_size = (int(intr[4]), int(intr[5]))
	
	K = correct_intrinsic_scale(raw_K, resize[0]/ raw_size[0], resize[1]/ raw_size[1]) if resize else raw_K
	img_size = resize if resize else raw_size
	
	node = ImageNode(
		img_name, rgb_img, depth_img, desc, 
		None, np.zeros((1, 3)), np.zeros((1, 4)), K, img_size,
		str(scene_path/img_name), str(scene_path/img_name)
	)
	node.set_raw_intrinsics(raw_K, raw_size)
	
	T_cam_world = convert_vec_to_matrix(pose[4:], pose[:4], mode='wxyz')
	T_world_cam = np.linalg.inv(T_cam_world)
	trans, quat = convert_matrix_to_vec(T_world_cam, 'xyzw')
	node.set_pose_gt(trans, quat)

	return node

def main():
	parser = argparse.ArgumentParser(description="KeyFrame Selection Evaluation")
	parser.add_argument("--config", help="path to config file")
	parser.add_argument("--dataset_dir", required=True, help="Dataset root directory")
	parser.add_argument("--keyframe_dir", required=True, help="Keyframe directory")
	parser.add_argument("--keyframe_selector", required=True, choices=["full_kf", "pose_density", "feature", "landmark"])
	parser.add_argument("--split", required=True, choices=["train", "val", "test"])
	parser.add_argument("--image_match_models", nargs="+", default=["duster"], choices=available_models)
	parser.add_argument("--pose_solver", default="essentialmatrixmetricmean", choices=available_solvers)
	parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
	parser.add_argument("--image_size", type=int, nargs="+", help="Resizing dimensions")
	parser.add_argument("--out_dir", default="results", help="Output directory")
	parser.add_argument("--debug", action="store_true", help="Enable debug mode")
	args = parser.parse_args()
	
	eval(args)

if __name__ == "__main__":
	main()
