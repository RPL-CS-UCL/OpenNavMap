"""
Usage: python demo_vpr.py \
--method=cosplace --backbone=ResNet18 --descriptors_dimension=512 \
--database_folder=/Titan/dataset/data_litevloc/anymal_ops_mos/map/map_rgb \
--queries_folder=/Titan/dataset/data_litevloc/anymal_ops_mos/sample_obs/obs_rgb/ \
--no_labels --image_size 200 200 \
--num_preds_to_save 3 --log_dir anymal_ops_mos
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

import time
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from colorama import Fore, Back, Style

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

import parser
from dataloader import TestDataset
from utils.utils_vpr_method import *

def extract_descriptors(model, test_ds, args):
    """Extract and return all descriptors from the test dataset."""
    with torch.inference_mode():
        # Extract database descriptors
        logging.debug("Extracting database descriptors for evaluation/testing")
        database_subset_ds = Subset(test_ds, list(range(test_ds.num_database)))
        database_dataloader = DataLoader(
            dataset=database_subset_ds,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
        )
        all_descriptors = np.empty(
            (len(test_ds), args.descriptors_dimension), dtype="float32"
        )
        for images, indices, image_name in tqdm(database_dataloader):
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors

        # Extract query descriptors
        logging.debug(
            "Extracting queries descriptors for evaluation/testing using batch size 1"
        )
        queries_subset_ds = Subset(
            test_ds,
            list(range(test_ds.num_database, test_ds.num_database + test_ds.num_queries)),
        )
        queries_dataloader = DataLoader(
            dataset=queries_subset_ds, num_workers=args.num_workers, batch_size=1
        )
        for images, indices, image_name in tqdm(queries_dataloader):
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors

    queries_descriptors = all_descriptors[test_ds.num_database :]
    database_descriptors = all_descriptors[: test_ds.num_database]
    return queries_descriptors, database_descriptors

def predict(test_ds, vpr_model, match_model, args):
    queries_descriptors, database_descriptors = extract_descriptors(vpr_model, test_ds, args)
    queries_image_names = test_ds.queries_image_names
    database_image_names = test_ds.database_image_names
    print("Number of database_descriptors: ", len(database_descriptors))
    print("Number of queries_descriptors: ", len(queries_descriptors))

    results_dict = defaultdict(list)
    running_time = []
    match_model.initialize_model(database_descriptors, recall_values=3)
    for query_idx, desc in enumerate(queries_descriptors):
        start_time = time.time()
        recall_preds, pred, score = match_model.match(desc.reshape(1, -1))
        est_time = time.time() - start_time
        running_time.append(est_time)
        
        results_dict[queries_image_names[query_idx]] = [database_image_names[i] for i in recall_preds]

    avg_runtime = running_time[0] if len(running_time) == 1 else np.mean(running_time)
    return results_dict, avg_runtime

def save_submission(results_dict: dict, output_path: Path):
    results = np.empty((0, 2), dtype=object)
    for query_image_name, database_image_names in results_dict.items():
        vec = np.empty((1, 2), dtype=object)
        vec[0, 0], vec[0, 1] = query_image_name, database_image_names[0]
        results = np.vstack((results, vec))
        
    np.savetxt(output_path, results, fmt='%s %s')

def save_predictions(results_dict: dict, test_ds, log_dir):
    """Save visualizations of predictions."""    
    for query_idx, (query_image_name, database_image_names) in enumerate(results_dict.items()):
        query_path = [os.path.join(test_ds.queries_folder, query_image_name)]
        database_paths = [os.path.join(test_ds.database_folder, name) for name in database_image_names]
        image_paths = query_path + database_paths
        save_visualization(log_dir, query_idx, image_paths, [1] * len(image_paths))

def eval(args):
    ##### Dataloader
    test_ds = TestDataset(
        database_folder=args.database_folder,
        queries_folder=args.queries_folder,
        image_size=args.image_size
    )

    ##### Prediction
    output_root = Path(args.out_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "runtime_results.txt", "w") as f:
        vpr_model = initialize_vpr_model(args.method, args.backbone, args.descriptors_dimension, args.device)
        match_model = initialize_match_model(args.match_model)
        results_dict, avg_runtime = predict(test_ds, vpr_model, match_model, args)
        print(Fore.GREEN + f"Running VPR Method {args.method} with Match Method {args.match_model}" + Style.RESET_ALL)

        # Save runtimes to txt
        print(args.method, args.match_model, avg_runtime)
        runtime_str = f"{args.method}_{args.match_model}: {avg_runtime:.3f}s"
        f.write(runtime_str + "\n")
        tqdm.write(runtime_str)

        # Save predictions to txt per scene
        log_dir = Path(output_root / f"{args.method}_{args.match_model}")
        log_dir.mkdir(parents=True, exist_ok=True)
        database_name = args.database_folder.split('out_map_')[-1]
        query_name = args.queries_folder.split('out_map_')[-1]
        save_submission(results_dict, log_dir / f"submission_{database_name}_{query_name}.txt")
        if args.debug:
            Path(log_dir / f"preds").mkdir(parents=True, exist_ok=True)
            save_predictions(results_dict, test_ds, log_dir)

def main():
    args = parser.parse_arguments()
    eval(args)

if __name__ == "__main__":
    main()
