import os
import argparse
import time
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../Semantic-SAM'))
from semantic_sam import prepare_image, build_semantic_sam, SemanticSamAutomaticMaskGenerator

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

def plot_results(masks, image_ori, save_path='../vis/', suffix=''):
    """Plot segmentation masks and save results with optional suffix."""
    if os.path.isdir(save_path):
        im_name = f'example{suffix}.png'
    else:
        base = os.path.basename(save_path).split('.')[0]
        im_name = f'{base}_example{suffix}.png'
        save_path = os.path.dirname(save_path)
    
    fig = plt.figure()
    plt.imshow(image_ori)

    show_anns(masks)
    fig.canvas.draw()
    plt.savefig(os.path.join(save_path, im_name), bbox_inches='tight')
    plt.close()

    plt.figure()
    image_bg = image_ori.copy()
    image_bg.fill(0)
    plt.imshow(image_bg)
    show_anns(masks)
    fig.canvas.draw()
    plt.savefig(os.path.join(save_path, im_name.replace('example', 'mask')), bbox_inches='tight')
    plt.close()

def process_masks(masks, image_shape, dilation_kernel=5, min_area=50):
    """
    Process masks by dilating existing regions and segmenting unlabeled areas.
    
    Args:
        masks: List of mask dictionaries.
        image_shape: (height, width) of the image.
        dilation_kernel: Kernel size for dilation.
        min_area: Minimum area for new regions.
    
    Returns:
        List of processed masks.
    """
    processed = []
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation_kernel, dilation_kernel))
    
    # Dilate each mask
    for mask in masks:
        if mask['area'] < min_area:
            continue

        seg = mask['segmentation'].astype(np.uint8)
        dilated = cv2.dilate(seg, kernel)
        dilated_mask = dilated.astype(bool)
        rows, cols = np.where(dilated_mask)
        if len(rows) == 0:
            continue
        y_min, y_max = np.min(rows), np.max(rows)
        x_min, x_max = np.min(cols), np.max(cols)
        bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]
        processed.append({
            'segmentation': dilated_mask,
            'area': dilated_mask.sum(),
            'bbox': bbox,
            'category': 'unknown',
            'dyna_prob': 0.5
        })
    
    # Find unsegmented regions
    # if processed:
    #     combined_mask = np.logical_or.reduce([m['segmentation'] for m in processed])
    # else:
    #     combined_mask = np.zeros(image_shape, dtype=bool)
    # unsegmented = ~combined_mask
    
    # TODO(gogojjh): not segment unknown regions
    # Segment unlabeled regions
    # num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(unsegmented.astype(np.uint8), connectivity=4)
    # for label in range(1, num_labels):
    #     area = stats[label, cv2.CC_STAT_AREA]
    #     if area < min_area:
    #         continue
        
    #     component_mask = (labels == label)
    #     rows, cols = np.where(component_mask)
    #     y_min, y_max = np.min(rows), np.max(rows)
    #     x_min, x_max = np.min(cols), np.max(cols)
    #     bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]
    #     processed.append({
    #         'segmentation': component_mask,
    #         'area': area,
    #         'bbox': bbox
    #     })
    
    return processed

def resize_processed_masks(processed_masks, ori_image, old_size, new_size):
    """
    Resize processed masks with scaling factors and regenerate properties.
    
    Args:
        processed_masks: List of mask dictionaries with 'segmentation', 'area', 'bbox', 'dyna_prob'
        sx: Horizontal scale factor
        sy: Vertical scale factor
    
    Returns:
        List of resized mask dictionaries with updated properties
    """
    sx, sy = new_size[1] / old_size[1], new_size[0] / old_size[0]

    resized_masks = []
    for mask in processed_masks:
        orig_mask = mask['segmentation'].astype(np.uint8)       
        resized_mask = cv2.resize(orig_mask, (new_size[1], new_size[0]), interpolation=cv2.INTER_NEAREST)
        resized_mask = resized_mask.astype(bool)
        rows, cols = np.where(resized_mask)
        if len(rows) == 0:
            continue  # Skip empty masks
            
        y_min, y_max = np.min(rows), np.max(rows)
        x_min, x_max = np.min(cols), np.max(cols)
        new_bbox = [
            int(x_min/sx),  # Scale coordinates back to original space
            int(y_min/sy),
            int((x_max - x_min)/sx),
            int((y_max - y_min)/sy)
        ]
        
        new_area = int(mask['area'] * sx * sy)
        resized_masks.append({
            'segmentation': resized_mask,
            'area': new_area,
            'bbox': new_bbox,
            'category': mask['category'],
            'dyna_prob': mask['dyna_prob']  # Preserve original probability
        })
    
    resized_image = cv2.resize(ori_image, (new_size[1], new_size[0]), interpolation=cv2.INTER_NEAREST)
    
    return resized_masks, resized_image

def main():
    parser = argparse.ArgumentParser(description='Run Semantic SAM segmentation with mask processing.')
    parser.add_argument('--ckpt', type=str, 
                        default='swinl_only_sam_many2many.pth, swint_only_sam_many2many.pth',
                        help='Path to checkpoint file')
    parser.add_argument('--image_folder', type=str,
                        default='s00000/seq1',
                        help='Path to input image folder')
    parser.add_argument('--save_folder', type=str, default='vis',
                        help='Path to save output masks')
    parser.add_argument('--dilation_kernel', type=int, default=5,
                        help='Kernel size for dilating masks (default: 5).')
    parser.add_argument('--min_area', type=int, default=50,
                        help='Minimum area for new regions (default: 50).')
    args = parser.parse_args()

    os.makedirs(args.save_folder, exist_ok=True)
    semantic_sam = build_semantic_sam(model_type='L', ckpt=args.ckpt)
    mask_generator = SemanticSamAutomaticMaskGenerator(semantic_sam, level=[2, 3])

    for idx, img_name in enumerate(sorted(os.listdir(args.image_folder))):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(args.image_folder, img_name)
            original_image, input_image = prepare_image(img_path)

            img = np.asarray(Image.open(img_path).convert('RGB'))
            size0 = [img.shape[0], img.shape[1]] # HxW
            size1 = [original_image.shape[0], original_image.shape[1]]

            # Generate and save original masks
            start_time = time.time()
            masks = mask_generator.generate(input_image)
            print(f"Processing {img_name}: {time.time() - start_time:.3f}s")
            
            save_path = os.path.join(args.save_folder, f"mask_{os.path.splitext(img_name)[0]}")
            plot_results(masks, original_image, save_path=save_path)
            
            # Process masks and save results
            processed_masks = process_masks(masks, original_image.shape[:2],
                                            dilation_kernel=args.dilation_kernel, 
                                            min_area=args.min_area)
            scaled_masks, scaled_image = resize_processed_masks(processed_masks, original_image, size1, size0)
            plot_results(scaled_masks, scaled_image, save_path=save_path, suffix='.processed')
            
            height, width = scaled_masks[0]['segmentation'].shape
            static_mask = np.zeros((height, width), dtype=bool)
            static_mask.fill(True)

            np.save(save_path + '_init_masks.npy', {
                'instance_masks': scaled_masks,
                'static_mask': static_mask
            })

if __name__ == "__main__":
    main()
