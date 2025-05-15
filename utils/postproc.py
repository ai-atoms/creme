'''
python3
creme utils.postproc module
utility functions for postprocessing data
'''

import json
import logging
import os
import random
import sys
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from pycocotools import mask as mask_utils

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.utils.visualizer import ColorMode, Visualizer


def load_metrics_as_pandas(json_path):
    # Initialize a list to store the filtered JSON objects
    lines = []

    # Open the file and read it line by line
    with open(json_path, "r") as f:
        for line in f:
            # Check if the line contains the keyword "bbox/AP"
            if "bbox/AP" in line:
                print(line)  # This will print the JSON string

                # Convert the line (a JSON string) to a dictionary
                dict_line = json.loads(line.strip())  # strip() to remove any leading/trailing whitespace

                # Append the dictionary to the list
                lines.append(dict_line)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(lines)
    print (df.head())
    return df


def plot_metrics_3x3(json_path):
    directory = os.path.dirname(json_path) + "/"
    df = load_metrics_as_pandas(json_path)

    # List of columns to plot
    columns_to_plot = ['total_loss','mask_rcnn/false_positive', 'mask_rcnn/false_negative', 
                       'mask_rcnn/accuracy', 'segm/AP', 'bbox/AP', 
                       'loss_mask', 'loss_rpn_loc', 'validation_loss']

    # Drop rows with any NaN values in the columns of interest
    df.dropna(subset=['iteration'] + columns_to_plot, inplace=True)

    # Create a 3x3 grid of plots using matplotlib
    fig, axes = plt.subplots(3, 3, figsize=(12, 8))
    axes = axes.flatten()  # Flatten the 2D array of axes to iterate over it

    # Plot each column vs 'iteration'
    for i, col in enumerate(columns_to_plot):
        ax = axes[i]
        ax.scatter(df['iteration'], df[col], label=col)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(col)
        ax.set_title(f'{col} vs Iteration')
        ax.legend()

    # Hide any empty subplots (if less than 9 plots)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.savefig(directory+'metrics.png')


def plot_metrics_3x3b(json_path):
    directory = os.path.dirname(json_path) + "/"
    df = load_metrics_as_pandas(json_path)

    # List of columns to plot
    columns_to_plot = ['total_loss','mask_rcnn/false_positive', 'mask_rcnn/false_negative', 
                       'mask_rcnn/accuracy', 'segm/AP', 'segm/AP50', 
                       'loss_mask', 'loss_rpn_loc', 'validation_loss']

    # Drop rows with any NaN values in the columns of interest
    df.dropna(subset=['iteration'] + columns_to_plot, inplace=True)

    # Create a 3x3 grid of plots using matplotlib
    fig, axes = plt.subplots(3, 3, figsize=(12, 8))
    axes = axes.flatten()  # Flatten the 2D array of axes to iterate over it

    # Plot each column vs 'iteration'
    for i, col in enumerate(columns_to_plot):
        ax = axes[i]
        ax.scatter(df['iteration'], df[col], label=col)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(col)
        ax.set_title(f'{col} vs Iteration')
        ax.legend()

    # Hide any empty subplots (if less than 9 plots)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.savefig(directory+'metrics_b.png')


def single_overlay(target_image, model_path, cfg, threshold=0.8, save=True):
    # get the model and set paths
    cfg.MODEL.WEIGHTS = f'{model_path}'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (threshold)  # testing threshold
    filename = os.path.splitext(os.path.basename(target_image))[0]

    # create a predictor
    predictor = DefaultPredictor(cfg)
    start = datetime.now()

    # load the image 
    im = cv2.imread(target_image)

    # prediction
    outputs = predictor(im)

    contours = []
    for pred_mask in outputs['instances'].pred_masks:
        mask = pred_mask.cpu().numpy().astype('uint8') # convert (True, False) masks to 8-bit numpy array 
        contour, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)   
        try: 
            contours.append(contour[0]) # take the first element which is the array of contour points
        except:
            pass

    # get the original img, plot contours and save image
    original_img = cv2.imread(target_image)
    image_with_contours = original_img.copy()

    for contour in contours:
        cv2.drawContours(image_with_contours, [contour], -1, (0, 255, 255), 2) 

    if save:
        cv2.imwrite(f"output/{filename}.png", image_with_contours)
    else:
        cv2.imshow('Contours overlay', image_with_contours)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("Time needed for running:", datetime.now() - start)
    

def batch_overlay(target_folder, model_path, cfg, threshold=0.8, save=True):
    for filename in os.listdir(target_folder):
        if filename.endswith(".png"): # check extension
            print (f'Processing {filename}...')
            image_path = os.path.join(target_folder, filename)
            single_overlay(image_path, model_path, cfg, threshold, save)


def show_min_validation_loss(json_path):
    # Load the JSON file into a pandas DataFrame
    df = load_metrics_as_pandas(json_path)

    # Check if the required column exists
    if 'validation_loss' not in df.columns:
        raise ValueError("'validation_loss' column not found in the dataset.")

    # List of columns to plot
    coi = ['iteration', 
            'total_loss','mask_rcnn/false_positive', 'mask_rcnn/false_negative', 
            'mask_rcnn/accuracy', 'segm/AP', 'bbox/AP', 
            'loss_mask', 'loss_rpn_loc', 'validation_loss']

    # Drop rows with any NaN values in the columns of interest
    df.dropna(subset=coi, inplace=True)
    
    # Sort the DataFrame by 'validation_loss' in ascending order
    sorted_df = df.sort_values(by='validation_loss', ascending=True)
    sorted_df = sorted_df[coi]
    
    # Display the top 10 entries
    print(sorted_df.head(10))


def load_coco_annotations(json_path):
    """Load COCO annotations from a JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def get_random_sample(coco_data, num_samples=8):
    """Select a random sample of images from COCO annotations."""
    images = coco_data["images"]
    return random.sample(images, min(num_samples, len(images)))


def get_annotations_for_image(image_id, coco_data):
    """Get annotations for a given image ID."""
    return [ann for ann in coco_data["annotations"] if ann["image_id"] == image_id]


def plot_sample_images(coco_path, img_folder, num_samples=8, save=True):
    """Plot a random sample of images with segmentation overlays."""
    # Load COCO data
    coco_data = load_coco_annotations(coco_path)
    
    # Get random images
    sample_images = get_random_sample(coco_data, num_samples)

    # Set up plot
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))  # 2 rows, 4 columns
    axes = axes.flatten()
    
    for idx, image_info in enumerate(sample_images):
        img_path = os.path.join(img_folder, image_info["file_name"])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get annotations for the image
        annotations = get_annotations_for_image(image_info["id"], coco_data)

        # Plot image
        ax = axes[idx]
        ax.imshow(image)

        # Overlay segmentation masks
        for ann in annotations:
            if "segmentation" in ann:
                segs = ann["segmentation"]
                for seg in segs:
                    poly = np.array(seg).reshape(-1, 2)
                    ax.plot(poly[:, 0], poly[:, 1], 'r-', linewidth=2)

        ax.axis("off")
        ax.set_title(f"ID: {image_info['id']}")

    # Hide unused subplots
    for i in range(idx + 1, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    if save:
        plt.savefig('output/samples.png')
    else:
        plt.show()


if __name__ == '__main__':
    print ('[creme] utils.postproc module loaded;')