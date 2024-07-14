"""
This file works as a pre-processing function file to make sure that the images can be used for cloud deployed models.
"""

import os
from PIL import Image

def preprocess_image(input_path, output_path):
    with Image.open(input_path) as img:
        # Convert the image to RGB mode
        # This will drop the alpha channel if it exists, which was always causing the error
        rgb_img = img.convert('RGB')
        
        # Save the image
        rgb_img.save(output_path)

def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"processed_{filename}")
            preprocess_image(input_path, output_path)
            print(f"Processed: {filename}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess images in a directory")
    parser.add_argument("input_directory", help="Directory containing the images to process")
    parser.add_argument("output_directory", help="Directory to save the processed images")
    args = parser.parse_args()

    process_directory(args.input_directory, args.output_directory)
