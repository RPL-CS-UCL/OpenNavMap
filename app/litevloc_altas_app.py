import os
import sys
import argparse
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import faiss
from loguru import logger
from PIL import Image
import pandas as pd
import pyproj
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import folium
from scipy.spatial.transform import Rotation

sys.path.append(os.path.join(os.path.dirname(__file__), '../../VPR-methods-evaluation'))
import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from altas_dataset import AltasDataset
from python.utils.utils_image import load_rgb_image
from litevloc_altas import load_megaloc_model, rerank, local_loc, MIN_MATCHED_KPTS, RELIABLE_CONF_THRESHOLD
from python.utils.utils_image_matching_method import initialize_img_matcher
from python.utils.utils_map_merging import initialize_pose_estimator
from python.utils.utils_geom import compute_pose_error

VPR_MODEL = None
DATABASE_DS = None
DATABASE_DESCRIPTORS = None
FAISS_INDEX = None
ARGS = None
TRANSFORM = None
IMG_MATCHER = None
POSE_ESTIMATOR = None

def ecef_to_latlon(x, y, z):
    """Converts ECEF coordinates to latitude and longitude."""
    try:
        ecef = pyproj.CRS("EPSG:4978")  # ECEF coordinate system
        wgs84 = pyproj.CRS("EPSG:4326")  # WGS84 lat/lon
        transformer = pyproj.Transformer.from_crs(ecef, wgs84, always_xy=True)
        lon, lat, _ = transformer.transform(x, y, z)
        return lat, lon
    except Exception as e:
        logger.error(f"Failed to convert ECEF to Lat/Lon: {e}")
        return None, None

def setup(args):
    """Initializes models and loads the database descriptors."""
    global VPR_MODEL, DATABASE_DS, DATABASE_DESCRIPTORS, FAISS_INDEX, ARGS, TRANSFORM, IMG_MATCHER, POSE_ESTIMATOR
    
    ARGS = args
    logger.info("Setting up the localization pipeline...")

    resize = [224, 224]
    transformations = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize(size=resize, antialias=True),
    ]
    TRANSFORM = transforms.Compose(transformations)

    # Load the VPR model
    VPR_MODEL = load_megaloc_model(ARGS.device)
    DATABASE_DS = AltasDataset(
        database_folder=ARGS.database_folder,
        image_size=resize,
        seq_len=1,
    )
    logger.info(f"Database loaded: {DATABASE_DS}")

    if ARGS.database_descriptors_path and os.path.exists(ARGS.database_descriptors_path):
        logger.info(f"Loading precomputed database descriptors from {ARGS.database_descriptors_path}")
        DATABASE_DESCRIPTORS = np.load(ARGS.database_descriptors_path)
    else:
        logger.info(f"Extracting descriptors from {len(DATABASE_DS)} database images...")
        with torch.inference_mode():
            DATABASE_DESCRIPTORS = np.empty((len(DATABASE_DS), 8448), dtype="float32")
            full_dataloader = DataLoader(
                dataset=DATABASE_DS, 
                num_workers=ARGS.num_workers, 
                batch_size=ARGS.batch_size
            )
            for images, indices in tqdm(full_dataloader, desc="Extracting database descriptors"):
                B, S, C, H, W = images.shape            
                descriptors = VPR_MODEL(images.to(ARGS.device))            
                if ARGS.device == "cuda":
                    torch.cuda.synchronize()
                DATABASE_DESCRIPTORS[indices.numpy()[:, -1], :] = descriptors.cpu().numpy()
        logger.info(f"Database descriptors extracted: {DATABASE_DESCRIPTORS.shape}")
        if ARGS.database_descriptors_path:
            np.save(ARGS.database_descriptors_path, DATABASE_DESCRIPTORS)

    logger.info("Building FAISS index for database descriptors...")
    FAISS_INDEX = faiss.IndexFlatL2(DATABASE_DESCRIPTORS.shape[1])
    FAISS_INDEX.add(DATABASE_DESCRIPTORS)
    
    # Load the image matcher
    if ARGS.matcher:
        logger.info(f"Loading image matcher: {ARGS.matcher}")
        IMG_MATCHER = initialize_img_matcher(ARGS.matcher, ARGS.device, ARGS.n_kpts)

    # Load the pose estimator
    if ARGS.pose_estimator:
        logger.info(f"Loading pose estimator: {ARGS.pose_estimator}")
        POSE_ESTIMATOR = initialize_pose_estimator(ARGS.pose_estimator, ARGS.device)

    logger.info("Setup complete. Application is ready.")

def localize_image(query_img_path):
    """
    Takes a user-uploaded image, performs global localization, and returns the results.
    """
    if VPR_MODEL is None or FAISS_INDEX is None:
        raise RuntimeError("Application is not initialized. Please run setup first.")

    query_img_pil = Image.open(query_img_path).convert("RGB")
    query_tensor = TRANSFORM(query_img_pil).unsqueeze(0).unsqueeze(0)

    ##### Global Localization #####
    with torch.inference_mode():
        query_descriptor = VPR_MODEL(query_tensor.to(ARGS.device))
        if ARGS.device == "cuda":
            torch.cuda.synchronize()
        query_descriptor = query_descriptor.cpu().numpy()

    _, predictions = FAISS_INDEX.search(query_descriptor, ARGS.recall_k)
    predictions = predictions[0]
    best_pred_idx = predictions[0]
    T_query_est_coarse = DATABASE_DS.get_image_pose(best_pred_idx)

    ##### Reranking #####
    if ARGS.matcher and IMG_MATCHER:
        logger.info(f"Reranking and verifying with {ARGS.matcher}...")
        predictions, num_matched_kpts = rerank(IMG_MATCHER, query_img_path, DATABASE_DS, predictions, ARGS)
        if predictions and num_matched_kpts[0] > MIN_MATCHED_KPTS:
            best_pred_idx = predictions[0]
            T_query_est_coarse = DATABASE_DS.get_image_pose(best_pred_idx)
        else:
            T_query_est_coarse = None
    else:
        num_matched_kpts = [0] * len(predictions)
        logger.info("Skipping reranking.")

    if T_query_est_coarse is None:
        m = None
        for idx in range(0, len(DATABASE_DS.database_poses), 10):
            db_pose_matrix = DATABASE_DS.database_poses[idx]
            db_x, db_y, db_z = db_pose_matrix[:3, 3]
            db_latitude, db_longitude = ecef_to_latlon(db_x, db_y, db_z)
            if db_latitude is not None and db_longitude is not None:
                if m is None:
                    m = folium.Map(location=[db_latitude, db_longitude], zoom_start=18)
                folium.Marker(
                    [db_latitude, db_longitude],
                    popup=f"DB Location {idx}",
                    icon=folium.Icon(color="green"),
                    z_index_offset=0,
                ).add_to(m)
        map_html = m._repr_html_() if m is not None else "<div>Map could not be generated.</div>"
        
        coords_str = f"The query is out of the premapped regions. Maximum matched KPts: {num_matched_kpts[0]}"
        best_match_path = DATABASE_DS.get_image_path(best_pred_idx)
        best_match_img = Image.open(best_match_path).convert("RGB")
        logger.warning(f"Not pass geometric verification")
        return best_match_img, coords_str, map_html

    ##### Local Localization #####
    conf = 0.0
    T_query_est_fine = T_query_est_coarse

    if ARGS.pose_estimator and POSE_ESTIMATOR:
        est_T, conf = local_loc(
            POSE_ESTIMATOR, query_img_path, DATABASE_DS, predictions[0], query_descriptor, DATABASE_DESCRIPTORS, ARGS
        )
        ##### DEBUG(gogojjh): the computation of confidence is sometimes zero
        # if est_T is not None and conf > RELIABLE_CONF_THRESHOLD:
        #####
        if est_T is not None:
            T_query_est_fine = est_T
        else:
            logger.warning(f"Local localization failed with confidence: {conf:.2f}")

    try:
        query_data = utils.parse_image_name(os.path.basename(query_img_path))
        T_query_gt = np.eye(4)
        T_query_gt[:3, 3] = utils.get_ecef_coords(
            query_data['easting'], query_data['northing'], int(query_data['zone_number']),
            query_data['latitude'], query_data['longitude']
        )
        r = Rotation.from_euler('zyx', [query_data['heading'], query_data['pitch'], query_data['roll']], degrees=True)
        T_query_gt[:3, :3] = r.as_matrix()
        trans_err_coarse, rot_err_coarse = compute_pose_error(T_query_gt, T_query_est_coarse)
        trans_err_fine, rot_err_fine = compute_pose_error(T_query_gt, T_query_est_fine)
    except Exception as e:
        trans_err_coarse, rot_err_coarse = -1.0, -1.0
        trans_err_fine, rot_err_fine = -1.0, -1.0
        
    best_match_path = DATABASE_DS.get_image_path(best_pred_idx)
    best_match_img = Image.open(best_match_path).convert("RGB")
    
    x, y, z = T_query_est_fine[:3, 3]
    latitude, longitude = ecef_to_latlon(x, y, z)
    if latitude is not None and longitude is not None:
        m = folium.Map(location=[latitude, longitude], zoom_start=18)
        for idx in range(0, len(DATABASE_DS.database_poses), 10):
            db_pose_matrix = DATABASE_DS.database_poses[idx]
            db_x, db_y, db_z = db_pose_matrix[:3, 3]
            db_latitude, db_longitude = ecef_to_latlon(db_x, db_y, db_z)
            if db_latitude is not None and db_longitude is not None:
                folium.Marker(
                    [db_latitude, db_longitude],
                    popup=f"DB Location {idx}",
                    icon=folium.Icon(color="green"),
                    z_index_offset=0,
                ).add_to(m)

        folium.Marker(
            [latitude, longitude],
            popup="Estimated Query Location",
            icon=folium.Icon(color="red"),
            z_index_offset=1000,
        ).add_to(m)
        map_html = m._repr_html_()

        coords_str = (
            f"Latitude: {latitude:.6f}, Longitude: {longitude:.6f}\n"
            f"Coarse Pose Error: {trans_err_coarse:.2f}m/{rot_err_coarse:.2f}deg\n"
            f"Fine Pose Error: {trans_err_fine:.2f}m/{rot_err_fine:.2f}deg\n"
            f"Maximum matched KPts: {num_matched_kpts[0]}\n"
            f"Confidence: {conf:.3f}"
        )
    else:
        coords_str = "Could not determine coordinates."
        m = None
        for idx in range(0, len(DATABASE_DS.database_poses), 10):
            db_pose_matrix = DATABASE_DS.database_poses[idx]
            db_x, db_y, db_z = db_pose_matrix[:3, 3]
            db_latitude, db_longitude = ecef_to_latlon(db_x, db_y, db_z)
            if db_latitude is not None and db_longitude is not None:
                if m is None:
                    m = folium.Map(location=[db_latitude, db_longitude], zoom_start=18)
                folium.Marker(
                    [db_latitude, db_longitude],
                    popup=f"DB Location {idx}",
                    icon=folium.Icon(color="green"),
                    z_index_offset=0,
                ).add_to(m)
        map_html = m._repr_html_() if m is not None else "<div>Map could not be generated.</div>"

    return best_match_img, coords_str, map_html

def main():
    parser = argparse.ArgumentParser(description='Gradio App for LiteVLoc Atlas')
    parser.add_argument('--assets_folder', type=str, required=True, help='Path to assets folder')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--database_folder', type=str, required=True, help='Path to database folder')
    parser.add_argument('--database_descriptors_path', type=str, required=True, help='Path to precomputed database descriptors .npy file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--share', action='store_true', help='Enable Gradio sharing')
    parser.add_argument('--recall_k', type=int, default=10, help='Number of top matches to return for reranking and localization')
    parser.add_argument('--matcher', type=str, default=None, help='Image matcher for reranking (e.g., superglue, loftr, master)')
    parser.add_argument('--n_kpts', type=int, default=2048, help='Max number of keypoints for image matcher')
    parser.add_argument('--pose_estimator', type=str, default=None, help='Pose estimator for local localization (e.g., mast3r, posecnn)')
    args = parser.parse_args()

    setup(args)

    example_images = []
    if os.path.isdir(args.assets_folder):
        image_files = sorted([
            f for f in os.listdir(args.assets_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        example_images = [os.path.join(args.assets_folder, f) for f in image_files[:5]]
    
    with gr.Blocks() as demo:
        gr.Markdown("# LiteVLoc Altas: Visual Localization with the Database: " + args.dataset_name)
        gr.Markdown("Upload an image to find its location on the map.")
        
        with gr.Row():
            with gr.Column():
                query_image_input = gr.Image(type="filepath", label="Upload Query Image")
                submit_button = gr.Button("Localize Image")
            with gr.Column():
                best_match_output = gr.Image(type="pil", label="Best Match from Database")
                coordinates_output = gr.Textbox(label="Estimated Coordinates", lines=5)
        
        gr.Examples(
            examples=example_images,
            inputs=query_image_input,
            label="Example Images",
        )
        
        with gr.Row():
            map_output_html = gr.HTML(label="Estimated Location on Map")
            
        submit_button.click(
            fn=localize_image,
            inputs=query_image_input,
            outputs=[best_match_output, coordinates_output, map_output_html]
        )
        
    demo.launch(share=args.share)

if __name__ == "__main__":
    main()
