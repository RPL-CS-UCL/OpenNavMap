import time
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

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

def plot_results(outputs, image_ori, save_path='../vis/'):
    """
    plot input image and its reuslts
    """
    if os.path.isdir(save_path):
        image_ori_name = 'input.png'
        im_name = 'example.png'
    else:
        image_ori_name = os.path.basename(save_path).split('.')[0] + '_input.png'
        im_name = os.path.basename(save_path).split('.')[0]+ '_example.png'
        save_path = os.path.dirname(save_path)
        
    if not os.path.exists(save_path):
        os.mkdir(save_path)       
        
    fig = plt.figure()
    plt.imshow(image_ori)
    plt.savefig(os.path.join(save_path, image_ori_name))
    show_anns(outputs)
    fig.canvas.draw()
    im = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.savefig(os.path.join(save_path, im_name))
    plt.close()

    fig = plt.figure()
    image_bg = image_ori.copy()
    image_bg.fill(0)
    plt.imshow(image_bg)
    show_anns(outputs)
    fig.canvas.draw()
    im = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.savefig(os.path.join(save_path, im_name.replace('example', 'mask')))
    plt.close()

    return im

if __name__ == "__main__":
    ckpt = '/Rocket_ssd/torch/hub/checkpoints/swinl_only_sam_many2many.pth'
    image_folder = '/Rocket_ssd/dataset/data_litevloc/map_free_eval/ucl_campus_aria/ucl_campus_P000_P000/map_free_eval/test/s00000/seq1'
    save_folder = 'vis'

    semantic_sam = build_semantic_sam(model_type='L', ckpt=ckpt) # model_type: 'L' / 'T', depends on your checkpint
    mask_generator = SemanticSamAutomaticMaskGenerator(semantic_sam, level=[2,3])

    # Iterate through all images in the folder
    for img_name in sorted(os.listdir(image_folder)):
        if img_name.endswith('.jpg') or img_name.endswith('.png'):  # Process only image files
            img_path = os.path.join(image_folder, img_name)  # Full path to the image

            # Prepare the image
            original_image, input_image = prepare_image(image_pth=img_path)

            # Generate masks
            start_time = time.time()
            masks = mask_generator.generate(input_image)
            print(f"Semantic SAM costs for {img_name}: {time.time() - start_time:.3f}s")

            # Save the results
            save_path = os.path.join(save_folder, f"mask_{img_name}")
            plot_results(masks, original_image, save_path=save_path)