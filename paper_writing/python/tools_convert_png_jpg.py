import os
from PIL import Image

def convert_png_to_jpg(directory):
    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"The directory '{directory}' does not exist.")
        return

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is a PNG image
        if filename.endswith(".png"):
            # Construct the full file path
            file_path = os.path.join(directory, filename)
            
            # Open the image
            with Image.open(file_path) as img:
                # Convert the image to RGB mode (necessary for JPEG)
                rgb_img = img.convert('RGB')
                
                # Construct the output file path with .jpg extension
                output_path = os.path.splitext(file_path)[0] + ".jpg"
                
                # Save the image in JPEG format
                rgb_img.save(output_path, "JPEG")
                print(f"Converted '{file_path}' to '{output_path}'")

if __name__ == "__main__":
    # Prompt the user to input the directory path
    datapath = input("Enter the directory path containing PNG images: ")
    
    # Convert all PNG images to JPG
    convert_png_to_jpg(datapath)
