'''
python3 
[creme] evaluating a pre-trained model
'''

import json
import os

import cv2
import numpy as np
import pandas as pd
import torch
from detectron2.config import get_cfg
from detectron2.data.datasets import load_coco_json
from detectron2.engine import DefaultPredictor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from core.models import MyTrainer, get_train_cfg_r50a

# Load dataset (ground truth)
ROOT_DIR = 'data/ds25s/fov200nm/'
ANNOTATION_PATH = f'{ROOT_DIR}clean.json'
IMAGE_DIR = f'{ROOT_DIR}imgs'
coco_gt = COCO(ANNOTATION_PATH)
dataset_dicts = load_coco_json(ANNOTATION_PATH, IMAGE_DIR)

# Configure model
job_name = 'ds25d_r50a_190325'
threshold = 0.90
dilation = True # 1px edge thickness for FG %

cfg = get_train_cfg_r50a(job_name)
cfg.OUTPUT_DIR = f'output/{job_name}'
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (threshold)  # testing threshold

predictor = DefaultPredictor(cfg)


def compute_foreground_percentage(binary_mask):
    total_area = binary_mask.shape[0] * binary_mask.shape[1]  # Total pixels in image
    foreground_area = np.sum(binary_mask > 0)  # Count nonzero pixels
    return (foreground_area / total_area) * 100


def evaluate_foreground_difference(image, ground_truth_annotations, predictor):
    
    # working on the model's predictions ...
    height, width = image.shape[:2]
    binary_pred_mask = np.zeros((height, width), dtype=np.uint8)
    outputs = predictor(image)
    
    contours_pred = []
    if outputs["instances"].has("pred_masks"):
        for pred_mask in outputs["instances"].pred_masks:
            mask = pred_mask.cpu().numpy().astype('uint8')  # Convert boolean mask to uint8
            contour, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            if contour:
                contours_pred.append(contour[0])

    # draw predicted contours
    for contour in contours_pred:
        cv2.drawContours(binary_pred_mask, [contour], -1, 255, thickness=cv2.FILLED)

    # include a dilation
    if dilation:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        dilate = cv2.dilate(binary_pred_mask, kernel, iterations=1)
    else: 
        dilate = binary_pred_mask

    fg_pred = compute_foreground_percentage(dilate)

    # extra layers
    n_objects_pred = len(contours_pred)
    ellipse_data = []
    
    for i, contour in enumerate(contours_pred):
        # Fit an ellipse to the contour
        if len(contour) >= 5:  # fitEllipse requires at least 5 points
            ellipse = cv2.fitEllipse(contour)
            center, axes, angle = ellipse
            major_axis_length = max(axes)
            minor_axis_length = min(axes)

            # Append the data to the list
            ellipse_data.append({
                'major_axis': major_axis_length,
                'minor_axis': minor_axis_length,
            })
        
    idf = pd.DataFrame(ellipse_data)
    major_ax_pred, minor_ax_pred = idf.mean()

    # working on the ground-truth reference ...
    binary_gt_mask = np.zeros((height, width), dtype=np.uint8)
    
    contours_gt = []
    for annotation in ground_truth_annotations:
        if "segmentation" in annotation:
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(annotation["segmentation"][0]).reshape(-1, 1, 2)], 255)
            contour, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            if contour:
                contours_gt.append(contour[0])

    # draw ground-truth contours
    for contour in contours_gt:
        cv2.drawContours(binary_gt_mask, [contour], -1, 255, thickness=cv2.FILLED)

    fg_gt = compute_foreground_percentage(binary_gt_mask)

    # extra layer
    n_objects_gt = len(contours_gt)
    ellipse_data2 = []
    
    for i, contour in enumerate(contours_gt):
        # Fit an ellipse to the contour
        if len(contour) >= 5:  # fitEllipse requires at least 5 points
            ellipse = cv2.fitEllipse(contour)
            center, axes, angle = ellipse
            major_axis_length = max(axes)
            minor_axis_length = min(axes)

            # Append the data to the list
            ellipse_data2.append({
                'major_axis': major_axis_length,
                'minor_axis': minor_axis_length,
            })
        
    jdf = pd.DataFrame(ellipse_data2)
    major_ax_gt, minor_ax_gt = jdf.mean()

    # computing differences
    if fg_gt > 0:
        fg_diff = abs(fg_pred - fg_gt) # Relative error
    else:
        fg_diff = 0 if fg_pred == 0 else np.nan  # Handle edge case when GT is empty


    return [
        fg_pred, # "FG% Predicted"
        fg_gt, # "FG% Ground Truth":
        fg_diff, # "Difference (%)": 
        n_objects_pred, # Number of objects Predicted
        n_objects_gt, # Number of objects Ground Truth
        major_ax_pred,
        major_ax_gt, 
        minor_ax_pred,
        minor_ax_gt,
    ]


results = []
for dataset_dict in dataset_dicts:  # Loop through dataset
    image_path = dataset_dict["file_name"]
    image = cv2.imread(image_path)

    ground_truth_annotations = dataset_dict["annotations"]

    result = evaluate_foreground_difference(image, ground_truth_annotations, predictor)
    # -- sanity check
    # print(f"Image: {image_path}")
    # print(result)
    results.append(result)


col_names = ['fg_pred', 'fg_gt', 'fg_diff',
              'n_objects_pred', 'n_objects_gt',
              'major_ax_pred', 'major_ax_gt',
              'minor_ax_pred', 'minor_ax_gt']

df = pd.DataFrame(results, columns=col_names)
print (df.mean())