'''
python3
creme utils.dataset module
utility functions for data management
'''

import json
import os
import re
from pathlib import Path

import cv2
import numpy as np


def natural_sort_key(filename):
    # Extract numbers from the filename for sorting
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else 0
    

def initialize_coco():
    # For a dataset with 1 class; to be adapted;
    return {
        "info": {
            "description": "ds25s",
            "version": "0.0.1",
            "year": 2025,
            "contributor": "atoms-ai",
            "date_created": "2025-05-14"
        },
        "licenses": [
            {
                "id": 1,
                "name": "MIT",
                "url": "github.com/ai-atoms/creme"
            }
        ],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "hcloop",  # Category for masks1
                "supercategory": "hcloop"
            }
        ]
    }


def process_mask(image_id, file_name, annotations, mask_dir, category_id):
    # Load the binary mask
    binary_mask = cv2.imread(os.path.join(mask_dir, file_name), cv2.IMREAD_GRAYSCALE)
    
    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract image dimensions
    height, width = binary_mask.shape
    
    # Add image info to the COCO structure
    image_info = {
        "id": image_id,
        "width": width,
        "height": height,
        "file_name": file_name,
        "license": 1,
        "flickr_url": "",
        "coco_url": "",
        "date_captured": "2025-05-14"
    }
    
    # Add each contour to the annotations list
    annotation_id = len(annotations) + 1
    for contour in contours:
        # Approximate the contour to reduce complexity (optional)
        contour = cv2.approxPolyDP(contour, epsilon=1.5, closed=True)  # Force closure

        # Flatten the contour points and force closure
        segmentation = contour.flatten().tolist()
        if len(segmentation) < 6:  # Skip invalid small segments
            continue

        # Ensure closure: first and last coordinates should match
        if segmentation[:2] != segmentation[-2:]:
            segmentation.extend(segmentation[:2])  # Append first point at the end

        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,  # Use the passed category_id
            "segmentation": [segmentation],
            "area": area,
            "bbox": [x, y, w, h],
            "iscrowd": 0
        }
        annotations.append(annotation)
        annotation_id += 1
    
    return image_info


def process_directory(masks1_dir, output_json):
    # For a dataset with 1 class
    coco_data = initialize_coco()

    # Get and sort the list of files using the natural sorting function
    files = [f for f in os.listdir(masks1_dir) if f.endswith('.png')]
    files.sort(key=natural_sort_key)

    # Process each mask in the sorted directory
    for file_name in files:
        # Extract the image_id from the filename (e.g., "1.png" -> image_id = 1)
        image_id = int(re.search(r'\d+', file_name).group())

        # Add image information using masks1
        image_info = process_mask(image_id, file_name, coco_data['annotations'], masks1_dir, category_id=1)
        coco_data['images'].append(image_info)

    # Save the COCO data to the specified JSON output file
    with open(output_json, 'w') as json_file:
        json.dump(coco_data, json_file, indent=4)

    print(f"COCO data generated and saved to {output_json}.")


def clean_invalid_segmentations(json_file, output_file):
    with open(json_file, 'r') as f:
        coco_data = json.load(f)

    valid_annotations = []
    removed_annotation_ids = []
    image_annotation_count = {}

    # First pass: Filter out invalid segmentations and count annotations per image
    for annotation in coco_data.get("annotations", []):
        segm = annotation.get("segmentation", [])
        image_id = annotation["image_id"]

        # Check for valid polygon format: list of lists, each with at least six coordinates
        if isinstance(segm, list) and all(isinstance(poly, list) and len(poly) >= 6 for poly in segm):
            valid_annotations.append(annotation)
            image_annotation_count[image_id] = image_annotation_count.get(image_id, 0) + 1
        # If using RLE format, it should have "counts" and "size"
        elif isinstance(segm, dict) and "counts" in segm and "size" in segm:
            valid_annotations.append(annotation)
            image_annotation_count[image_id] = image_annotation_count.get(image_id, 0) + 1
        else:
            removed_annotation_ids.append(annotation["id"])

    # Identify images with fewer than 2 annotations
    invalid_image_ids = {img_id for img_id, count in image_annotation_count.items() if count < 2}

    # Filter annotations again to remove those linked to images with < 2 objects
    final_annotations = [ann for ann in valid_annotations if ann["image_id"] not in invalid_image_ids]

    # Filter images as well
    final_images = [img for img in coco_data.get("images", []) if img["id"] not in invalid_image_ids]

    # Update the COCO data
    coco_data["annotations"] = final_annotations
    coco_data["images"] = final_images

    # Save the cleaned data
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=4)

    print(f"Total annotations removed: {len(removed_annotation_ids)}")
    print(f"Removed annotation IDs: {removed_annotation_ids}")
    print(f"Total images removed: {len(invalid_image_ids)}")
    print(f"Cleaned data saved to {output_file}")


def get_image_values(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None, None
    
    # Calculate mean and standard deviation
    mean = np.mean(image)
    stddev = np.std(image)
    
    return mean, stddev


def get_folder_values(folder_path):
    means = []
    stddevs = []
    
    # List all files in the directory
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
            image_path = os.path.join(folder_path, filename)
            mean, stddev = get_image_values(image_path)
            if mean is not None and stddev is not None:
                means.append(mean)
                stddevs.append(stddev)
    
    if means and stddevs:
        avg_mean = np.mean(means)
        avg_stddev = np.mean(stddevs)
        print(f"Average Mean Pixel Value: {avg_mean:.2f}")
        print(f"Average Standard Deviation: {avg_stddev:.2f}")
    else:
        print("No valid images found in the folder.")
    
    return avg_mean, avg_stddev


def get_size_distribution(json_path):
    with open(json_path, "r") as f:
        coco_data = json.load(f)

    small, medium, large = 0, 0, 0

    for ann in coco_data["annotations"]:
        area = ann["area"]  # COCO defines area in pixel^2
        if area < 32**2:
            small += 1
        elif area < 96**2:
            medium += 1
        else:
            large += 1  # This is what should contribute to Large AR

    print(f"Small: {small}, Medium: {medium}, Large: {large}")


def image_fold_ij(image_path, path_out, file_name, folds_i, folds_j, resize=True):
    # Split and save i-j folds

    im = cv2.imread(image_path)
    height, width = im.shape[:2]
    fold_height = height // folds_i
    fold_width = width // folds_j
    
    # Create output directory if it doesn't exist
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    
    for i in range(folds_i):
        for j in range(folds_j):
            # Define the coordinates of the current fold
            y_start = i * fold_height
            y_end = y_start + fold_height
            x_start = j * fold_width
            x_end = x_start + fold_width
            
            # Extract the fold
            roi = im[y_start:y_end, x_start:x_end]          

            if resize:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray_roi = cv2.resize(gray_roi, (512, 512))
                roi = gray_roi
            
            # Save the grayscale fold
            output_filename = os.path.join(path_out, f'{file_name}_{i*folds_j + j + 1}.png')
            cv2.imwrite(output_filename, roi)


def folder_fold_ij(folder_path, path_out, folds_i, folds_j, resize=True):
    path_in = Path(folder_path)
    
    for i, file in enumerate(path_in.glob("*")):
        # this loop iterate over the images from the folder_path
        if file.suffix.lower() in [".png", ".tif", ".tiff"]:
            file_name = file.stem
            image_fold_ij(file, path_out, file_name, folds_i, folds_j, resize)


def rename_files_in_folder(folder_path, start_num):
    # Get the list of all files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and not f.startswith('.')]
    
    # Sort files using the custom sort key based on numerical order
    files.sort(key=natural_sort_key) # ATTENTION: works only with numerical filenames!

    # Use a temporary name during renaming to avoid overwriting issues
    temp_suffix = "_temp"

    # First pass: rename files to temporary names
    for i, filename in enumerate(files):
        old_file = os.path.join(folder_path, filename)
        temp_name = f"{start_num + i}{temp_suffix}.png"
        temp_file = os.path.join(folder_path, temp_name)
        os.rename(old_file, temp_file)

    # Second pass: rename from temporary names to final names
    for i in range(len(files)):
        temp_file = os.path.join(folder_path, f"{start_num + i}{temp_suffix}.png")
        new_file = os.path.join(folder_path, f"{start_num + i}.png")
        os.rename(temp_file, new_file)

    print(f"Files renamed successfully starting from {start_num}.")


if __name__ == '__main__':
    print ('[creme] utils.dataset module loaded;')