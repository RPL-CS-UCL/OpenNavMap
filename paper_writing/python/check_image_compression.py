from PIL import Image
import os

# Path to JPEG file
# data_folder = '/Rocket_ssd/dataset/data_litevloc/map_free_eval/mapfree/map_free_eval/test/s00460/seq1'
# data_folder = '/Rocket_ssd/dataset/data_litevloc/map_free_eval/ucl_campus_aria/map_free_eval/test/s00100/seq1'
data_folder = '/Rocket_ssd/dataset/data_litevloc/map_free_eval/matterport3d/map_free_eval/test/s00000/seq1'

# Get all image files in the folder
image_files = [f for f in os.listdir(data_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Process each image
for image_file in image_files:
    image_path = os.path.join(data_folder, image_file)
    # print(f"\nProcessing {image_file}:")

    # 1. Get compressed size
    compressed_size = os.path.getsize(image_path)  # in bytes

    # 2. Load image
    with Image.open(image_path) as img:
        width, height = img.size

    # 3. Uncompressed size in memory
    uncompressed_size = width * height * 3  # RGB = 3 bytes per pixel

    # 4. Calculate compression ratio
    compression_ratio = compressed_size / uncompressed_size

    # 5. Print results
    # print(f"Image size: {width}x{height}")
    # print(f"Compressed size: {compressed_size / 1024:.2f} KB")
    # print(f"Uncompressed size: {uncompressed_size / 1024:.2f} KB")
    # print(f"Compression ratio: {compression_ratio:.3f}")

    print(f"Compress size: {compressed_size/1024:2f} KB, Compression ratio: {compression_ratio:.3f}")