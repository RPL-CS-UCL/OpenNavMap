import os
import sys
from glob import glob
from collections import defaultdict

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from scipy.spatial.transform import Rotation

sys.path.append(os.path.join(os.path.dirname(__file__), '../../VPR-methods-evaluation'))
import utils

def read_images_paths(dataset_folder):
    """Finds image paths in a folder, using a pre-generated index if available."""
    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"Folder {dataset_folder} does not exist")
    
    file_with_paths = dataset_folder + "_images_paths.txt"
    if os.path.exists(file_with_paths):
        with open(file_with_paths, "r") as file:
            images_paths = [os.path.join(dataset_folder, p) for p in file.read().splitlines()]
        if images_paths and not os.path.exists(images_paths[0]):
            raise FileNotFoundError(f"Image path {images_paths[0]} from index file is invalid.")
    else:
        print(f"Searching images in {dataset_folder} with glob()")
        images_paths = sorted(glob(f"{dataset_folder}/**/*", recursive=True))
        images_paths = [p for p in images_paths if os.path.isfile(p) and os.path.splitext(p)[1].lower() in [".jpg", ".jpeg", ".png"]]
        if not images_paths:
            raise FileNotFoundError(f"Directory {dataset_folder} contains no JPEG or PNG images.")
    return images_paths

class AltasDataset(data.Dataset):
    def __init__(self, database_folder, image_size=None, seq_len=1):
        """Dataset for loading database images and their poses as 4x4 matrices."""
        super().__init__()
        
        self.seq_len = seq_len
        
        raw_database_paths = read_images_paths(database_folder)
        self.database_paths, self.data_groups = self.parse_and_sort_paths(raw_database_paths)
        self.images_paths = list(self.database_paths)
        self.num_database = len(self.database_paths)
        
        db_poses_list = []
        for path in self.database_paths:
            data = utils.parse_image_name(path)
            T = np.eye(4)
            
            T[:3, 3] = utils.get_ecef_coords(
                data['easting'], data['northing'], int(data['zone_number']), 
                data['latitude'], data['longitude']
            )
            
            r = Rotation.from_euler('zyx', [data['heading'], data['pitch'], data['roll']], degrees=True)
            T[:3, :3] = r.as_matrix()
            db_poses_list.append(T)
            
        self.database_poses = np.array(db_poses_list)
        
        transformations = [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        if image_size:
            transformations.append(transforms.Resize(size=image_size, antialias=True))
        self.transform = transforms.Compose(transformations)
    
    def parse_and_sort_paths(self, raw_database_paths):
        """Parses and sorts image paths by scene and image ID."""
        def group_and_sort(items):
            scene_to_items = defaultdict(list)
            for item in items:
                scene = item.get('scene') or os.path.basename(item['path'])
                scene_to_items[scene].append(item)
            return [item for scene in sorted(scene_to_items)
                    for item in sorted(scene_to_items[scene], key=lambda x: x.get('img_id') or 0)]
        
        parse_data = []
        for img_name in raw_database_paths:
            data = utils.parse_image_name(img_name) or {}
            parse_data.append({
                'path': img_name,
                'scene': data.get('scene'),
                'img_id': data.get('img_id')
            })
        
        data_groups = group_and_sort(parse_data)
        database_paths = [item['path'] for item in data_groups]
        
        return database_paths, data_groups
    
    def __getitem__(self, index):
        """Returns a sequence of images ending at the given index."""
        scene = self.data_groups[index]['scene']
        
        image_paths, indices = [], []
        start_index = index - self.seq_len + 1
        for idx in range(start_index, index + 1):
            if idx < 0 or self.data_groups[idx]['scene'] != scene:
                continue
            image_paths.append(self.images_paths[idx])
            indices.append(idx)
        
        if len(image_paths) < self.seq_len:
            image_paths = [image_paths[-1]] * (self.seq_len - len(image_paths)) + image_paths
            indices = [indices[-1]] * (self.seq_len - len(indices)) + indices
        
        imgs = []
        for path in image_paths:
            try:
                pil_img = Image.open(path).convert("RGB")
                imgs.append(self.transform(pil_img))
            except Exception as e:
                print(f"Error opening image {path}: {e}, using last valid image instead.")
                if imgs:
                    imgs.append(imgs[-1])
        
        return torch.stack(imgs), torch.tensor(indices)
    
    def __len__(self):
        return len(self.images_paths)
    
    def __repr__(self):
        return f"< AltasDataset: #database: {self.num_database} >"
    
    def get_image_path(self, index):
        return self.images_paths[index]

    def get_image_pose(self, index):
        """Returns the 4x4 pose matrix for the image at the given index."""
        return self.database_poses[index]

