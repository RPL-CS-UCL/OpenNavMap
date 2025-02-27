#!/bin/bash

export EGOBLUR_PATH="/Titan/code/robohike_ws/src/EgoBlur"
export FACE_MODEL_PATH="/Rocket_ssd/image_matching_model_weights/ego_blur_face.jit"
export LP_MODEL_PATH="/Rocket_ssd/image_matching_model_weights/ego_blur_lp.jit"
export DATA_PATH="/Titan/dataset/data_litevloc/data_icra2025/map_free_eval/ucl_campus/map_free_eval/"

# Create output root directory
mkdir -p "$DATA_PATH/test_blur"

# Process all scenes
for SCENE in "$DATA_PATH"/test/*/; do
    SCENE_NAME=$(basename "$SCENE")
    
    # Process both seq0 and seq1
    for SEQ in seq0 seq1; do
        INPUT_DIR="$DATA_PATH/test/$SCENE_NAME/$SEQ"
        OUTPUT_DIR="$DATA_PATH/test_blur/$SCENE_NAME/$SEQ"
        
        # Create output directory
        mkdir -p "$OUTPUT_DIR"
        
        # Process all frame_*.jpg images
        find "$INPUT_DIR" -name "frame_*.jpg" | while read -r INPUT_IMAGE; do
            # Get filename without path
            FILENAME=$(basename "$INPUT_IMAGE")
            
            # Create output path
            OUTPUT_IMAGE="$OUTPUT_DIR/$FILENAME"
            
            # Run processing script
            python "$EGOBLUR_PATH/script/demo_ego_blur.py" \
                --face_model_path "$FACE_MODEL_PATH" \
                --lp_model_path "$LP_MODEL_PATH" \
                --face_model_score_threshold 0.5 \
                --lp_model_score_threshold 0.5 \
                --input_image_path "$INPUT_IMAGE" \
                --output_image_path "$OUTPUT_IMAGE"
        done
    done
done
