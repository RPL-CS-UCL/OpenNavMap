
import os
import numpy as np
from glob import glob
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors

def read_poses(dataset_folder):
    pose_path = os.path.join(dataset_folder, 'poses_abs_gt.txt')

    if not os.path.exists(pose_path):
        print(f"File not found: {pose_path}")
        return None

    data_dict = {}
    with open(pose_path, 'r') as f:
        for line_id, line in enumerate(f):
            if line.startswith('#'):
                continue

            parts = line.strip().split()
            if 'jpg' in parts[0] or 'png' in parts[0]:
                # provide img_name
                img_name = parts[0]
                data = list(map(float, parts[1:]))
            else:
                # not provide img_name
                img_name = "seq/{frame_id:06d}.color.jpg".format(frame_id=line_id)
                data = list(map(float, parts))

            data_dict[img_name] = np.array(data)
            
    return data_dict


def read_image_names(dataset_folder):
    """Find images within 'dataset_folder'. If the file
    'dataset_folder'_images_paths.txt exists, read paths from such file.
    Otherwise, use glob(). Keeping the paths in the file speeds up computation,
    because using glob over large folders might be slow.
    
    Parameters
    ----------
    dataset_folder : str, folder containing JPEG images
    
    Returns
    -------
    images_paths : list[str], paths of JPEG images within dataset_folder
    """
    
    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"Folder {dataset_folder} does not exist")

    image_names = []
    print(f"Searching test images in {dataset_folder} with glob()")
    for path in sorted(glob(f"{dataset_folder}/seq/*.color.jpg", recursive=True)):
        image_name = path.split(f"{dataset_folder}/")[-1]
        image_names.append(image_name)
    
    return image_names

class TestDataset(data.Dataset):
    def __init__(self, database_folder, queries_folder, image_size=None, normalized=True):
        """Dataset with images from database and queries, used for validation and test.
        Parameters
        ----------
        dataset_folder : str, should contain the path to the val or test set,
            which contains the folders {database_folder} and {queries_folder}.
        database_folder : str, name of folder with the database.
        queries_folder : str, name of folder with the queries.
        """
        super().__init__()
        
        self.database_folder = database_folder
        self.database_image_names = read_image_names(database_folder)
        self.database_image_paths = [
            os.path.join(database_folder, name) for name in self.database_image_names
        ]
        self.database_poses = read_poses(database_folder)

        self.queries_folder = queries_folder
        self.queries_image_names = read_image_names(queries_folder)
        self.queries_image_paths = [
            os.path.join(queries_folder, name) for name in self.queries_image_names
        ]        
        self.queries_poses = read_poses(queries_folder)
    
        self.image_names = list(self.database_image_names) + list(self.queries_image_names)
        self.image_paths = list(self.database_image_paths) + list(self.queries_image_paths)

        self.num_database = len(self.database_image_names)
        self.num_queries = len(self.queries_image_names)
                
        transformations = [transforms.ToTensor()]
            
        if normalized:
            transformations.append(
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )

        if image_size:
            transformations.append(transforms.Resize(size=image_size, antialias=True))

        self.transform = transforms.Compose(transformations)
    
    def __getitem__(self, index):
        image_name = self.image_names[index]
        image_path = self.image_paths[index]
        pil_img = Image.open(image_path).convert("RGB")
        normalized_img = self.transform(pil_img)
        return normalized_img, index, image_name
    
    def __len__(self):
        return len(self.image_paths)
    
    def __repr__(self):
        return f"< #queries: {self.num_queries}; #database: {self.num_database} >"

