#!/usr/bin/env python

import os
import argparse
import cv2

def get_image_files(image_folder):
    # Accept common image extensions
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    files = [f for f in os.listdir(image_folder) if f.lower().endswith(valid_exts)]
    files = [f for f in files if 'vpr' in f]
    files.sort()
    return [os.path.join(image_folder, f) for f in files]

def images_to_video(image_folder, output_video_path, fps):
    image_files = get_image_files(image_folder)
    if not image_files:
        print(f"No images found in {image_folder}")
        return

    # Read first image to get size
    first_img = cv2.imread(image_files[0])
    if first_img is None:
        print(f"Failed to read {image_files[0]}")
        return
    height, width, layers = first_img.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to read {img_path}, skipping.")
            continue
        if (img.shape[0], img.shape[1]) != (height, width):
            img = cv2.resize(img, (width, height))
        video_writer.write(img)

    video_writer.release()
    print(f"Video saved to {output_video_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert a folder of images to an MP4 video.")
    parser.add_argument('--image_folder', '-i', required=True, help='Path to the folder containing images.')
    parser.add_argument('--output_video_path', '-o', required=True, help='Path to save the output MP4 video.')
    parser.add_argument('--fps', '-r', type=float, default=10.0, help='Frame rate (frames per second) for the video.')
    args = parser.parse_args()

    images_to_video(args.image_folder, args.output_video_path, args.fps)

if __name__ == "__main__":
    main()
