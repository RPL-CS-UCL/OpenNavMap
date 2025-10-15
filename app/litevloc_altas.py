import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import faiss
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation

# Add paths to third-party modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../../VPR-methods-evaluation/third_party/MegaLoc'))
from megaloc_model import MegaLocModel
sys.path.append(os.path.join(os.path.dirname(__file__), '../../VPR-methods-evaluation'))
import utils

# Add path to litevloc python modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from python.utils.utils_image_matching_method import initialize_img_matcher
from python.utils.utils_image import load_rgb_image
from python.utils.utils_map_merging import initialize_pose_estimator
from python.utils.utils_geom import compute_pose_error

# Import local dataset
from altas_dataset import AltasDataset

VISUALIZE = False
# Geometric Verification
MIN_MATCHED_KPTS = 100
# Local Localization
TRANS_THRESH_M, ROT_THRESH_DEG = 7.5, 90.0
N_IMG_LOCAL_LOC = 2
RELIABLE_CONF_THRESHOLD = 0.5

def load_megaloc_model(device='cuda'):
    """Load pre-trained MegaLoc model."""
    logger.info("Loading MegaLoc model...")
    model = MegaLocModel()
    model.load_state_dict(
        torch.hub.load_state_dict_from_url(
            "https://github.com/gmberton/MegaLoc/releases/download/v1.0/megaloc.torch", 
            map_location=torch.device(device)
        )
    )
    logger.info("MegaLoc model loaded successfully")
    return model.eval().to(device)

def global_loc(model, database_ds, args):
    """
    Extracts database and query descriptors, and performs global localization.
    """
    base_transformations = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    #### 1. Extract database descriptors ####
    if args.database_descriptors_path and os.path.exists(args.database_descriptors_path):
        logger.info(f"Loading precomputed database descriptors from {args.database_descriptors_path}")
        database_descriptors = np.load(args.database_descriptors_path)
    else:
        logger.info(f"Extracting descriptors from {len(database_ds)} database images...")
        with torch.inference_mode():
            all_descriptors = np.empty((len(database_ds), 8448), dtype="float32")
            full_dataloader = DataLoader(
                dataset=database_ds, 
                num_workers=args.num_workers, 
                batch_size=args.batch_size
            )
            for images, indices in tqdm(full_dataloader, desc="Extracting database descriptors"):
                B, S, C, H, W = images.shape            
                descriptors = model(images.to(args.device))            
                if args.device == "cuda":
                    torch.cuda.synchronize()
                all_descriptors[indices.numpy()[:, -1], :] = descriptors.cpu().numpy()
        database_descriptors = all_descriptors
        logger.info(f"Database descriptors extracted: {database_descriptors.shape}")
        if args.database_descriptors_path:
            np.save(args.database_descriptors_path, database_descriptors)

    #### 2. Load query images ####
    seq_len = len(args.img_files)
    assert len(args.img_files) == seq_len, "Number of query images must match sequence length"
    
    transformations = base_transformations.copy()
    if args.image_size:
        transformations.append(transforms.Resize(size=args.image_size, antialias=True))
    transform = transforms.Compose(transformations)
    
    imgs = []
    for img_file in args.img_files:
        if not os.path.exists(img_file):
            raise FileNotFoundError(f"Query image {img_file} does not exist")
        
        pil_img = Image.open(img_file).convert("RGB")
        normalized_img = transform(pil_img)
        imgs.append(normalized_img)
    
    query_images = torch.stack(imgs).unsqueeze(0)

    #### 3. Extract query descriptor ####
    with torch.inference_mode():
        descriptor = model(query_images.to(args.device))
        if args.device == "cuda":
            torch.cuda.synchronize()
        query_descriptor = descriptor.cpu().numpy()

    #### 4. Perform FAISS search ####
    faiss_index = faiss.IndexFlatL2(database_descriptors.shape[1])
    faiss_index.add(database_descriptors)
    
    _, predictions = faiss_index.search(query_descriptor, args.recall_k)
    
    return predictions[0], query_descriptor, database_descriptors

def rerank(img_matcher, query_img_path, database_ds, predictions, args):
    query_image = load_rgb_image(query_img_path).to(args.device)

    match_results = []
    for pred_idx in predictions:
        db_img_path = database_ds.get_image_path(pred_idx)
        db_image = load_rgb_image(db_img_path).to(args.device)
        with torch.no_grad():
            result = img_matcher(db_image, query_image)
        num_inliers = result['num_inliers']
        match_results.append((pred_idx, num_inliers))

    reranked_pairs = sorted(match_results, key=lambda x: x[1], reverse=True)
    reranked_predictions, sorted_kpts = zip(*reranked_pairs) if reranked_pairs else ([], [])
    
    return list(reranked_predictions), list(sorted_kpts)

def local_loc(pose_estimator, query_img_path, database_ds, db_idx, query_descriptor, database_descriptors, args):
    db_pose = database_ds.get_image_pose(db_idx)
    db_position = db_pose[:3, 3].reshape(1, -1)
    all_db_positions = database_ds.database_poses[:, :3, 3].astype(np.float32)
    
    # Retrieve DB images within translation and rotation thresholds
    faiss_index_pos = faiss.IndexFlatL2(3)
    faiss_index_pos.add(all_db_positions)
    _, _, indices = faiss_index_pos.range_search(db_position, TRANS_THRESH_M)
    candidate_indices = indices

    ##### Option 1: use rotation threshold to sort the candidates
    # filtered_db_indices = [
    #     idx for idx in candidate_indices
    #     if Rotation.from_matrix(db_pose[:3, :3].T @ database_ds.get_image_pose(idx)[:3, :3]).magnitude() * 180/np.pi <= ROT_THRESH_DEG
    # ]
    # if len(filtered_db_indices) <= 1:
    #     logger.warning(f"No images found within {TRANS_THRESH_M}m and {ROT_THRESH_DEG} degrees.")
    #     return None
    # db_indices = filtered_db_indices[:N_IMG_LOCAL_LOC]
    ##### Option 2: use query descriptor to sort the candidates
    db_descriptors = database_descriptors[candidate_indices]
    dists = np.linalg.norm(db_descriptors - query_descriptor, axis=1)
    db_indices = candidate_indices[np.argsort(dists)][:N_IMG_LOCAL_LOC]
    #################

    # Prepare DB and query data
    db_image_paths = [database_ds.get_image_path(idx) for idx in db_indices]
    db_images = [load_rgb_image(p, resize=(512, 288)).to(args.device) for p in db_image_paths]
    T_w_local = database_ds.get_image_pose(db_indices[0])
    db_poses_matrices = [
        torch.from_numpy(np.linalg.inv(T_w_local) @ database_ds.get_image_pose(idx)).float() 
        for idx in db_indices
    ]
    query_image = load_rgb_image(query_img_path, resize=(512, 288)).to(args.device)

    # Perform pose estimation
    logger.info(f"Performing local localization with {args.pose_estimator} using DB images {db_indices}...")
    est_opts = {'known_extrinsics': True, 'known_intrinsics': False, 'niter': 300}    
    result = pose_estimator(
        Path(args.database_folder),
        db_images, query_image,
        db_poses_matrices, None, None,
        est_opts
    )
    if VISUALIZE:
        pose_estimator.show_reconstruction()

    # Use confidence to check if the pose estimation is reliable
    top_k_matches = len(db_indices) # default: 2
    if hasattr(pose_estimator, 'get_minimum_spanning_tree'):
        msp_edges = pose_estimator.get_minimum_spanning_tree()
        weight_i, weight_j = pose_estimator.scene.weight_i, pose_estimator.scene.weight_j
        for edge in msp_edges:
            if edge[0] == top_k_matches or edge[1] == top_k_matches: # confidence of the query image
                edge_str = f"{edge[0]}_{edge[1]}"
                conf = (weight_i[edge_str].mean() * weight_j[edge_str].mean()).detach().cpu().item()

    # Get the pose estimation result
    im_pose = result.get("im_pose")
    if im_pose is None or np.isnan(im_pose).any():
        logger.error(f"{args.pose_estimator} - failed to estimate pose.")
        return None, conf

    T_w_query = T_w_local @ im_pose
    logger.info(f"Local localization successful.")

    return T_w_query, conf

def process_and_display_results(title, predictions, database_ds, T_query_gt, recall_k, file_handle=None, num_matched_kpts=None):
    """Process and display results of VPR."""

    logger.info(f"Top-{recall_k} Results ({title}):")
    if file_handle:
        file_handle.write(f"\nTop-{recall_k} ({title}):\n")

    for rank, idx in enumerate(predictions, 1):
        db_pose_matrix = database_ds.get_image_pose(int(idx))
        geo_distance = np.linalg.norm(T_query_gt[:3, 3] - db_pose_matrix[:3, 3])
        image_path = database_ds.get_image_path(int(idx))
        img_name = os.path.basename(image_path)
        
        gd_str = f", {geo_distance:.2f}m"
        if num_matched_kpts: gd_str += f", {num_matched_kpts[rank-1]}KPs"
        logger.info(f"R{rank}: ID={int(idx)}, {img_name}{gd_str}")
        if file_handle:
            file_handle.write(f"R{rank}: ID={int(idx)}, {image_path}{gd_str}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LiteVLoc Atlas - Global Localization with MegaLoc')
    parser.add_argument('--database_folder', type=str, required=True, help='Path to database folder')
    parser.add_argument('--img_files', type=str, nargs='+', required=True, help='Path(s) to query image file(s)')
    parser.add_argument('--recall_k', type=int, default=10, help='Number of top matches to return (R1-RK)')
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224], help='Image size (height, width)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for database descriptor extraction')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--database_descriptors_path', type=str, default=None, help='Path to precomputed database descriptors')
    parser.add_argument('--output_file', type=str, default=None, help='Optional output file to save results')
    parser.add_argument('--matcher', type=str, default=None, help='Image matcher for reranking (e.g., superglue, loftr, master)')
    parser.add_argument('--n_kpts', type=int, default=2048, help='Max number of keypoints for image matcher')
    parser.add_argument('--pose_estimator', type=str, default=None, help='Pose estimator for local localization (e.g., mast3r, posecnn)')

    args = parser.parse_args()
    seq_len = len(args.img_files)

    logger.remove()
    logger.add(sys.stdout, colorize=True, format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")

    #### Global Localization ####    
    vpr_model = load_megaloc_model(args.device)
    database_ds = AltasDataset(
        database_folder=args.database_folder,
        image_size=args.image_size,
        seq_len=seq_len,
    )
    logger.info(f"Database loaded: {database_ds}")
    predictions, query_descriptor, database_descriptors = global_loc(vpr_model, database_ds, args)
    T_query_est_coarse = database_ds.get_image_pose(predictions[0])
    del vpr_model

    #### Rerank using image matching and Verification ####
    if args.matcher:
        logger.info(f"Reranking and verifying with {args.matcher}...")
        img_matcher = initialize_img_matcher(args.matcher, args.device, args.n_kpts)
        predictions, num_matched_kpts = rerank(img_matcher, args.img_files[-1], database_ds, predictions, args)
        del img_matcher
        if predictions and num_matched_kpts[0] > MIN_MATCHED_KPTS:
            T_query_est_coarse = database_ds.get_image_pose(predictions[0])
        else:
            T_query_est_coarse = None
    else:
        num_matched_kpts = [0] * len(predictions)
        logger.info("Skipping reranking.")

    #### Local Localization ####
    T_query_est_fine = T_query_est_coarse
    if args.pose_estimator and T_query_est_coarse is not None:
        pose_estimator = initialize_pose_estimator(args.pose_estimator, args.device)
        est_T, conf = local_loc(pose_estimator, args.img_files[-1], database_ds, predictions[0], query_descriptor, database_descriptors, args)
        if est_T is not None and conf > RELIABLE_CONF_THRESHOLD: 
            T_query_est_fine = est_T
        del pose_estimator
    else:
        conf = 0.0

    #### Output results ####
    query_data = utils.parse_image_name(args.img_files[-1])
    T_query_gt = np.eye(4)
    T_query_gt[:3, 3] = utils.get_ecef_coords(
        query_data['easting'], query_data['northing'], int(query_data['zone_number']),
        query_data['latitude'], query_data['longitude']
    )
    r = Rotation.from_euler('zyx', [query_data['heading'], query_data['pitch'], query_data['roll']], degrees=True)
    T_query_gt[:3, :3] = r.as_matrix()  

    logger.info(f'Ground Truth Pose: {T_query_gt[:3, 3].T}')
    if T_query_est_fine is None:
        logger.info(f"The query is out of the premapped regions. Maximum match KPts: {num_matched_kpts[0]}")
        trans_err_coarse, rot_err_coarse = -1.0, -1.0
        trans_err_fine, rot_err_fine = -1.0, -1.0
    else:
        trans_err_coarse, rot_err_coarse = compute_pose_error(T_query_gt, T_query_est_coarse)
        trans_err_fine, rot_err_fine = compute_pose_error(T_query_gt, T_query_est_fine)
        logger.info(f'Coarse Estimated Pose: {T_query_est_coarse[:3, 3].T}')
        logger.info(f'Fine Estimated Pose: {T_query_est_fine[:3, 3].T}')

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'a') as f:
            f.write(f"Query: {', '.join(args.img_files)}\n")
            f.write(f"DB: {args.database_folder}\n")
            if args.matcher:
                process_and_display_results(
                    "VPR with Reranking", predictions, database_ds, T_query_gt, args.recall_k, f, num_matched_kpts
                )
            else:
                process_and_display_results(
                    "VPR wo Reranking", predictions, database_ds, T_query_gt, args.recall_k, f, num_matched_kpts
                )
            f.write(f"Coarse-to-Fine Pose Error: {trans_err_coarse:.2f}m/{rot_err_coarse:.2f}deg, {trans_err_fine:.2f}m/{rot_err_fine:.2f}deg\n")
            f.write(f"Confidence: {conf:.3f}\n")
        logger.info(f"Results saved to {output_path}")
    else:
        if args.matcher:    
            process_and_display_results(
                "VPR with Reranking", predictions, database_ds, T_query_gt, args.recall_k, num_matched_kpts
            )
        else:
            process_and_display_results(
                "VPR wo Reranking", predictions, database_ds, T_query_gt, args.recall_k, num_matched_kpts
            )
    logger.info(f"Coarse-to-Fine Pose Error: {trans_err_coarse:.2f}m/{rot_err_coarse:.2f}deg, {trans_err_fine:.2f}m/{rot_err_fine:.2f}deg")
    logger.info(f"Local localization with confidence: {conf:.3f}")
