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

    VPR_MODEL = load_megaloc_model(ARGS.device)

    DATABASE_DS = AltasDataset(
        database_folder=ARGS.database_folder,
        image_size=ARGS.image_size,
        seq_len=1,
    )
    logger.info(f"Database loaded: {DATABASE_DS}")

    if ARGS.database_descriptors_path and os.path.exists(ARGS.database_descriptors_path):
        logger.info(f"Loading precomputed database descriptors from {ARGS.database_descriptors_path}")
        DATABASE_DESCRIPTORS = np.load(ARGS.database_descriptors_path)
    else:
        raise FileNotFoundError("Database descriptors not found. Please generate them first using litevloc_altas.py")

    logger.info("Building FAISS index for database descriptors...")
    FAISS_INDEX = faiss.IndexFlatL2(DATABASE_DESCRIPTORS.shape[1])
    FAISS_INDEX.add(DATABASE_DESCRIPTORS)
    
    base_transformations = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if ARGS.image_size:
        base_transformations.append(transforms.Resize(size=ARGS.image_size, antialias=True))
    TRANSFORM = transforms.Compose(base_transformations)

    if ARGS.matcher:
        logger.info(f"Loading image matcher: {ARGS.matcher}")
        IMG_MATCHER = initialize_img_matcher(ARGS.matcher, ARGS.device, ARGS.n_kpts)

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
    query_tensor = TRANSFORM(query_img_pil).unsqueeze(0).unsqueeze(0).to(ARGS.device)

    ##### Global Localization #####
    with torch.no_grad():
        descriptor = VPR_MODEL(query_tensor)
        query_descriptor = descriptor.cpu().numpy()

    _, predictions = FAISS_INDEX.search(query_descriptor, ARGS.recall_k)
    predictions = predictions[0]

    T_query_est_coarse = DATABASE_DS.get_image_pose(predictions[0])
    best_pred_idx = predictions[0]

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
        coords_str = f"The query is out of the premapped regions. Maximum matched KPts: {num_matched_kpts[0]}"
        map_html = "<div>Map could not be generated.</div>"
        best_match_path = DATABASE_DS.get_image_path(best_pred_idx)
        best_match_img = Image.open(best_match_path).convert("RGB")
        return best_match_img, coords_str, map_html

    ##### Local Localization #####
    conf = 0.0
    T_query_est_fine = T_query_est_coarse

    if ARGS.pose_estimator and POSE_ESTIMATOR:
        est_T, conf = local_loc(POSE_ESTIMATOR, query_img_path, DATABASE_DS, predictions[0], query_descriptor, DATABASE_DESCRIPTORS, ARGS)
        if est_T is not None and conf > RELIABLE_CONF_THRESHOLD:
            T_query_est_fine = est_T

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

    best_match_path = DATABASE_DS.get_image_path(best_pred_idx)
    best_match_img = Image.open(best_match_path).convert("RGB")
    
    x, y, z = T_query_est_fine[:3, 3]
    latitude, longitude = ecef_to_latlon(x, y, z)
    if latitude is not None and longitude is not None:
        coords_str = (
            f"Latitude: {latitude:.6f}, Longitude: {longitude:.6f}\n"
            f"Coarse Pose Error: {trans_err_coarse:.2f}m/{rot_err_coarse:.2f}deg\n"
            f"Fine Pose Error: {trans_err_fine:.2f}m/{rot_err_fine:.2f}deg\n"
            f"Maximum matched KPts: {num_matched_kpts[0]}\n"
            f"Confidence: {conf:.3f}"
        )
        m = folium.Map(location=[latitude, longitude], zoom_start=18)
        folium.Marker([latitude, longitude], popup="Estimated Location").add_to(m)
        map_html = m._repr_html_()

    else:
        coords_str = "Could not determine coordinates."
        map_html = "<div>Map could not be generated.</div>"

    return best_match_img, coords_str, map_html

def main():
    parser = argparse.ArgumentParser(description='Gradio App for LiteVLoc Atlas')
    parser.add_argument('--database_folder', type=str, required=True, help='Path to database folder')
    parser.add_argument('--database_descriptors_path', type=str, required=True, help='Path to precomputed database descriptors .npy file')
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224], help='Image size (height, width)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--share', action='store_true', help='Enable Gradio sharing')
    parser.add_argument('--recall_k', type=int, default=10, help='Number of top matches to return for reranking and localization')
    parser.add_argument('--matcher', type=str, default=None, help='Image matcher for reranking (e.g., superglue, loftr, master)')
    parser.add_argument('--n_kpts', type=int, default=2048, help='Max number of keypoints for image matcher')
    parser.add_argument('--pose_estimator', type=str, default=None, help='Pose estimator for local localization (e.g., mast3r, posecnn)')
    args = parser.parse_args()

    setup(args)
    
    with gr.Blocks() as demo:
        gr.Markdown("# LiteVLoc Atlas: Visual Localization")
        gr.Markdown("Upload an image to find its location on the map.")
        
        with gr.Row():
            with gr.Column():
                query_image_input = gr.Image(type="filepath", label="Upload Query Image")
                submit_button = gr.Button("Localize Image")
            with gr.Column():
                best_match_output = gr.Image(type="pil", label="Best Match from Database")
                coordinates_output = gr.Textbox(label="Estimated Coordinates", lines=4)
        
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
