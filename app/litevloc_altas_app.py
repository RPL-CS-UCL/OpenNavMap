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

# --- Add paths to dependencies ---
sys.path.append(os.path.join(os.path.dirname(__file__), '../../VPR-methods-evaluation'))
import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# --- Import from existing scripts ---
from altas_dataset import AltasDataset
from python.utils.utils_image import load_rgb_image
from litevloc_altas import load_megaloc_model

# --- Global variables for the app ---
VPR_MODEL = None
DATABASE_DS = None
DATABASE_DESCRIPTORS = None
FAISS_INDEX = None
ARGS = None
TRANSFORM = None

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
    global VPR_MODEL, DATABASE_DS, DATABASE_DESCRIPTORS, FAISS_INDEX, ARGS, TRANSFORM
    
    ARGS = args
    logger.info("Setting up the localization pipeline...")

    # 1. Load VPR Model
    VPR_MODEL = load_megaloc_model(ARGS.device)

    # 2. Load Database
    DATABASE_DS = AltasDataset(
        database_folder=ARGS.database_folder,
        image_size=ARGS.image_size,
        seq_len=1,
    )
    logger.info(f"Database loaded: {DATABASE_DS}")

    # 3. Load or compute database descriptors
    if ARGS.database_descriptors_path and os.path.exists(ARGS.database_descriptors_path):
        logger.info(f"Loading precomputed database descriptors from {ARGS.database_descriptors_path}")
        DATABASE_DESCRIPTORS = np.load(ARGS.database_descriptors_path)
    else:
        raise FileNotFoundError("Database descriptors not found. Please generate them first using litevloc_altas.py")

    # 4. Build FAISS index
    logger.info("Building FAISS index for database descriptors...")
    FAISS_INDEX = faiss.IndexFlatL2(DATABASE_DESCRIPTORS.shape[1])
    FAISS_INDEX.add(DATABASE_DESCRIPTORS)
    
    # 5. Define image transformation
    base_transformations = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if ARGS.image_size:
        base_transformations.append(transforms.Resize(size=ARGS.image_size, antialias=True))
    TRANSFORM = transforms.Compose(base_transformations)

    logger.info("Setup complete. Application is ready.")

def localize_image(query_img_pil):
    """
    Takes a user-uploaded image, performs global localization, and returns the results.
    """
    if VPR_MODEL is None or FAISS_INDEX is None:
        raise RuntimeError("Application is not initialized. Please run setup first.")

    # 1. Preprocess query image
    query_tensor = TRANSFORM(query_img_pil.convert("RGB")).unsqueeze(0).unsqueeze(0).to(ARGS.device)

    # 2. Extract query descriptor
    with torch.no_grad():
        descriptor = VPR_MODEL(query_tensor)
        query_descriptor = descriptor.cpu().numpy()

    # 3. Perform FAISS search to find the best match
    _, predictions = FAISS_INDEX.search(query_descriptor, 1)
    best_pred_idx = predictions[0][0]

    # 4. Get results for the best match
    # Get the best matching image from the database
    best_match_path = DATABASE_DS.get_image_path(best_pred_idx)
    best_match_img = Image.open(best_match_path).convert("RGB")
    
    # Get the pose and convert to latitude and longitude
    pose_matrix = DATABASE_DS.get_image_pose(best_pred_idx)
    x, y, z = pose_matrix[:3, 3]
    latitude, longitude = ecef_to_latlon(x, y, z)

    if latitude is not None and longitude is not None:
        coords_str = f"Latitude: {latitude:.6f}, Longitude: {longitude:.6f}"
        
        # Create an interactive map with Folium and get its HTML representation
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
    args = parser.parse_args()

    # Initialize the models and data
    setup(args)
    
    # Create and launch the Gradio interface
    with gr.Blocks() as demo:
        gr.Markdown("# LiteVLoc Atlas: Visual Localization")
        gr.Markdown("Upload an image to find its location on the map.")
        
        with gr.Row():
            with gr.Column():
                query_image_input = gr.Image(type="pil", label="Upload Query Image")
                submit_button = gr.Button("Localize Image")
            with gr.Column():
                best_match_output = gr.Image(type="pil", label="Best Match from Database")
                coordinates_output = gr.Textbox(label="Estimated Coordinates")
        
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
