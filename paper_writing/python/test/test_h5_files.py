#!/usr/bin/env python

import os
import sys
import h5py
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Load and display H5 file contents')
    parser.add_argument('--h5_path', type=str, required=True,
                      help='Path to the H5 file')
    return parser.parse_args()

def print_h5_structure(name, obj):
    """Print the structure of the H5 file"""
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}")
        print(f"  Shape: {obj.shape}")
        print(f"  Dtype: {obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"Group: {name}")

def main():
    # Parse arguments
    args = parse_args()
    
    # Check if file exists
    if not os.path.exists(args.h5_path):
        print(f"Error: File {args.h5_path} does not exist")
        return

    try:
        # Open the H5 file
        with h5py.File(args.h5_path, 'r') as f:
            print("\nH5 File Structure:")
            print("-" * 50)
            f.visititems(print_h5_structure)
                    
            # Example of how to access data
            print("\nExample of accessing data:")
            print("-" * 50)
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    data = f[key][:]
                    print(f"\nDataset '{key}':")
                    print(f"First few elements: {data[:5] if len(data.shape) == 1 else data[:5, :5]}")
                    print(data)

                elif isinstance(f[key], h5py.Group):
                    print(f"\nGroup '{key}' contains:")
                    for subkey in f[key].keys():
                        print(f"  - {subkey}")

    except Exception as e:
        print(f"Error reading H5 file: {str(e)}")

if __name__ == '__main__':
    main()