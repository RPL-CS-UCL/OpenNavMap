# Long-Term Scene Change Detection
## :unicorn: Getting Started

### :hammer_and_wrench: Installation
```shell
conda install pytorch=2.0.1 torchvision=0.15.2 pytorch-cuda=11.8 numpy=1.24.3 -c pytorch -c nvidia # use the correct version of cuda for your system
python -m pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git'
pip install git+https://github.com/cocodataset/panopticapi.git
git clone https://github.com/UX-Decoder/Semantic-SAM
cd Semantic-SAM
python -m pip install -r requirements.txt
```

### Data Structure
# Scene Mask Processor Documentation

This repository contains tools for processing instance masks and generating static masks for scene understanding. The implementation uses NumPy for efficient array operations and supports dynamic probability updates for objects in a scene.

---

## Data Structures
###### 1. **Instance Masks**: Each instance mask represents an object in the scene and is stored as a dictionary with the following fields:

```python
instance_masks = []
instance_mask.append({
    'segmentation': dilated_mask,  # 2D boolean array (True = object present)
    'area': dilated_mask.sum(),    # Total area of the object (pixel count)
    'bbox': bbox,                  # Bounding box [x_min, y_min, width, height]
    'category': 'unknown',         # Object category (e.g., 'vehicle', 'pedestrian')
    'dyna_prob': 0.5                # Dynamic probability (0.0 to 1.0)
})
```

###### 2. **Static Mask**: The static mask represents the background or non-dynamic regions of the scene:
```python
static_mask = np.zeros((height, width), dtype=bool)
static_mask.fill(True)
```

###### 3. **Combined Mask**: The combined mask structure stores both instance masks and the static mask:
```python
mask = {
    'instance_masks': scaled_masks,  # List of processed instance masks
    'static_mask': static_mask       # 2D static mask
}
np.save('mask.npy', mask)
```

###### 4. Usage:
```python
masks = np.load(input_mask_path, allow_pickle=True)
instance_masks = masks.item().get('instance_masks', [])
static_mask = masks.item().get('static_mask', [])
```
