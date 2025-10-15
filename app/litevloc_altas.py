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
from python.utils.utils_geom import convert_vec_to_matrix

# Import local dataset
from altas_dataset import AltasDataset

MIN_MATCHED_KPTS = 100

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
    logger.info(f"Loaded {len(args.img_files)} query images with shape {query_images.shape}")

    #### 3. Extract query descriptor ####
    with torch.inference_mode():
        descriptor = model(query_images.to(args.device))
        if args.device == "cuda":
            torch.cuda.synchronize()
        query_descriptor = descriptor.cpu().numpy()

    #### 4. Perform FAISS search ####
    logger.info("Building FAISS index...")
    faiss_index = faiss.IndexFlatL2(database_descriptors.shape[1])
    faiss_index.add(database_descriptors)
    
    _, predictions = faiss_index.search(query_descriptor, args.recall_k)
    
    return predictions[0]

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

def get_default_intrinsics(image_size):
    W, H = image_size
    f = max(W, H) * 1.2
    K = torch.tensor([[f, 0, W/2], [0, f, H/2], [0, 0, 1]], dtype=torch.float32)
    im_size_tensor = torch.tensor([W, H], dtype=torch.float32)
    return {'K': K, 'im_size': im_size_tensor}

def local_loc(pose_estimator, query_img_path, database_ds, reranked_predictions, args):
    top_pred_idx = reranked_predictions[0]
    top_pred_pose = database_ds.get_image_pose(top_pred_idx)
    top_pred_position = top_pred_pose[:3, 3].reshape(1, -1)
    all_db_positions = database_ds.database_poses[:, :3, 3].astype(np.float32)
    
    # Retrieve DB images within translation and rotation thresholds
    faiss_index_pos = faiss.IndexFlatL2(3)
    faiss_index_pos.add(all_db_positions)
    top_pos = top_pred_position.astype(np.float32)
    _, nn_indices = faiss_index_pos.search(top_pos, len(all_db_positions))

    trans_thresh_m, rot_thresh_deg = 7.5, 60.0
    top_pose = database_ds.get_image_pose(top_pred_idx)
    db_indices_candidates = nn_indices[0]
    filtered_db_indices = []
    for idx in db_indices_candidates:
        pose = database_ds.get_image_pose(idx)
        trans = pose[:3, 3]
        rot = pose[:3, :3]
        translation_dist = np.linalg.norm(trans - top_pose[:3, 3])
        delta_rot = top_pose[:3, :3].T @ rot
        rot_angle = Rotation.from_matrix(delta_rot).magnitude() * (180.0 / np.pi)
        if translation_dist <= trans_thresh_m and rot_angle <= rot_thresh_deg:
            filtered_db_indices.append(idx)

    if len(filtered_db_indices) <= 1:
        logger.warning(f"No images found within {trans_thresh_m}m and {rot_thresh_deg} degrees. Using second-best reranked prediction as fallback.")
        return None

    db_indices = filtered_db_indices[:min(5, len(filtered_db_indices))]

    # Prepare DB and query data
    db_image_paths = [database_ds.get_image_path(idx) for idx in db_indices]
    db_images = [load_rgb_image(p).to(args.device) for p in db_image_paths]
    db_poses_matrices = [torch.from_numpy(database_ds.get_image_pose(idx)).float() for idx in db_indices]        

    query_image = load_rgb_image(query_img_path).to(args.device)

    # Perform pose estimation
    # logger.info(f"Performing local localization with {args.pose_estimator} using DB images {db_indices}...")
    # est_opts = {'known_extrinsics': False, 'known_intrinsics': False, 'resize': 512, 'niter': 300}    
    # result = pose_estimator(
    #     Path(args.database_folder),
    #     db_images,
    #     query_image,
    #     db_poses_matrices,
    #     None,
    #     None,
    #     est_opts
    # )
    # pose_estimator.show_reconstruction()
    # exit()

    # Visualize DB image positions and their heading (rotation), mark each with ID
    for p in db_image_paths:
        name_data = utils.parse_image_name(p.split("/")[-1])
        print(f"DB Image Name: {name_data['scene']}, {name_data['img_id']}, {name_data['easting']}, {name_data['northing']}, {name_data['height']}")
        print(f"               {name_data['heading']}, {name_data['pitch']}, {name_data['roll']}")

    import matplotlib.pyplot as plt
    def plot_db_images(db_image_paths):
        fig, axs = plt.subplots(1, len(db_image_paths), figsize=(4 * len(db_image_paths), 4))
        if len(db_image_paths) == 1:
            axs = [axs]
        for ax, img_path in zip(axs, db_image_paths):
            img = load_rgb_image(img_path).permute(1, 2, 0).cpu().numpy()
            ax.imshow(img)
            ax.set_title(img_path.split("/")[-1], fontsize=10)
            ax.axis('off')
        plt.suptitle('Top DB Images')
        plt.tight_layout()
        plt.show()

    plot_db_images(db_image_paths)
    exit()
    
    im_pose = result.get("im_pose")
    if im_pose is None or np.isnan(im_pose).any():
        logger.error(f"{args.pose_estimator} - failed to estimate pose.")
        return None
        
    logger.info(f"Local localization successful.")
    return im_pose

def process_and_display_results(title, predictions, database_ds, query_pose_gt_matrix, recall_k, file_handle=None, num_matched_kpts=None):
    logger.info(f"Top-{recall_k} Results ({title}):")
    if file_handle:
        file_handle.write(f"\nTop-{recall_k} ({title}):\n")

    for rank, idx in enumerate(predictions, 1):
        db_pose_matrix = database_ds.get_image_pose(int(idx))
        geo_distance = np.linalg.norm(query_pose_gt_matrix[:3, 3] - db_pose_matrix[:3, 3])
        image_path = database_ds.get_image_path(int(idx))
        img_name = os.path.basename(image_path)
        
        gd_str = f", {geo_distance:.2f}m"
        if num_matched_kpts:
            gd_str += f", {num_matched_kpts[rank-1]}kpts"
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
    query_valid = True

    #### Global Localization ####    
    vpr_model = load_megaloc_model(args.device)
    database_ds = AltasDataset(
        database_folder=args.database_folder,
        image_size=args.image_size,
        seq_len=seq_len,
    )
    logger.info(f"Database loaded: {database_ds}")
    predictions = global_loc(vpr_model, database_ds, args)
    query_pose_est_matrix = database_ds.get_image_pose(predictions[0])
    del vpr_model

    #### Rerank using image matching and Verification ####
    reranked_predictions = []
    if args.matcher:
        logger.info(f"Reranking and verifying with {args.matcher}...")
        img_matcher = initialize_img_matcher(args.matcher, args.device, args.n_kpts)
        reranked_predictions, num_matched_kpts = rerank(img_matcher, args.img_files[-1], database_ds, predictions, args)
        del img_matcher
        if not reranked_predictions or num_matched_kpts[0] < MIN_MATCHED_KPTS:
            query_valid = False
        else:
            query_pose_est_matrix = database_ds.get_image_pose(reranked_predictions[0])
    else:
        logger.info("Skipping reranking.")

    #### Local Localization ####
    if args.pose_estimator:
        if query_valid and reranked_predictions:
            pose_estimator = initialize_pose_estimator(args.pose_estimator, args.device)
            est_pose_matrix = local_loc(pose_estimator, args.img_files[-1], database_ds, reranked_predictions, args)
            if est_pose_matrix is not None:
                query_pose_est_matrix = est_pose_matrix
            del pose_estimator

    #### Output results ####
    query_data = utils.parse_image_name(args.img_files[-1])
    query_pose_gt_matrix = np.eye(4)
    query_pose_gt_matrix[:3, 3] = utils.get_ecef_coords(
        query_data['easting'], query_data['northing'], int(query_data['zone_number']),
        query_data['latitude'], query_data['longitude']
    )
    r = Rotation.from_euler('zyx', [query_data['heading'], query_data['pitch'], query_data['roll']], degrees=True)
    query_pose_gt_matrix[:3, :3] = r.as_matrix()
    
    error = np.linalg.norm(query_pose_est_matrix[:3, 3] - query_pose_gt_matrix[:3, 3])
    logger.info(f"Final Estimated Pose Error: {error:.2f}m")

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(f"Query: {', '.join(args.img_files)}\n")
            f.write(f"DB: {args.database_folder}\n")
            f.write(f"Final Pose Error: {error:.2f}m\n")
            
            process_and_display_results("Global Ranking", predictions, database_ds, query_pose_gt_matrix, args.recall_k, f)
            if reranked_predictions:
                process_and_display_results("Reranked", reranked_predictions, database_ds, query_pose_gt_matrix, args.recall_k, f, num_matched_kpts)
        logger.info(f"Results saved to {output_path}")
    else:
        process_and_display_results("Global Ranking", predictions, database_ds, query_pose_gt_matrix, args.recall_k)
        if reranked_predictions:
            process_and_display_results("Reranked", reranked_predictions, database_ds, query_pose_gt_matrix, args.recall_k, num_matched_kpts)