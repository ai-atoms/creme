'''
python3 
[creme] segments a batch of images using a given model; saves the data in analysis.csv
'''

import logging
import os
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.utils.visualizer import ColorMode, Visualizer

from core.models import get_train_cfg_r50a


# presets
job_name = 'ds25d_r50a_190325'
target_folder = Path("data/media/video3/base_aug/")
output_folder = target_folder / "masks"
os.makedirs(output_folder, exist_ok=True)
threshold = 0.80
save_masks = True

# backbone model
cfg = get_train_cfg_r50a(job_name)

# cfg presets
# -- cfg relative to the dataset
cfg.OUTPUT_DIR = f'output/{job_name}'
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (threshold)  # testing threshold

# create a predictor
predictor = DefaultPredictor(cfg)
start = datetime.now()
main_df = pd.DataFrame() # to store all analytical data


for i, file in enumerate(target_folder.glob("*.png")):
    # this loop opens the .png files from the val-folder, creates a dict with the file
    file = str(file)
    file_name = file.split("/")[-1]
    image_id = file_name.split(".png")[0]
    
    # -- preprocess
    im = cv2.imread(file) # simply load the image
    
    # prediction
    outputs = predictor(im)

    # get dimensions and create an empty mask
    height, width, channels = im.shape
    binary_mask = np.zeros((height, width, channels), dtype=np.uint8)
    
    contours = []
    for pred_mask in outputs['instances'].pred_masks:
        mask = pred_mask.cpu().numpy().astype('uint8') # convert (True, False) masks to 8-bit numpy array 
        contour, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)   
        try: 
            contours.append(contour[0]) # take the first element which is the array of contour points
        except:
            pass

    # plot contours and save image
    for contour in contours:
        cv2.drawContours(binary_mask, [contour], -1, (255,255,255), thickness=cv2.FILLED) 


    if save_masks:
        if contours: 
            # contours must not be empty
            cv2.imwrite(f"{output_folder}/{file_name}", binary_mask)
    else:
        cv2.imshow('Binary Mask', binary_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    
    # analysis section
    ellipse_data = []
    
    for i, contour in enumerate(contours):
        # Fit an ellipse to the contour
        if len(contour) >= 5:  # fitEllipse requires at least 5 points
            ellipse = cv2.fitEllipse(contour)
            center, axes, angle = ellipse
            major_axis_length = max(axes)
            minor_axis_length = min(axes)

            center_str = f"({center[0]:.3f} - {center[1]:.3f})" # tuple to str to avoid error when exp

            # Append the data to the list
            ellipse_data.append({
                'image_id': image_id,
                'id': i,
                'center': center_str,
                'major_axis': major_axis_length,
                'minor_axis': minor_axis_length,
                'angle': angle
            })
        
    df = pd.DataFrame(ellipse_data)
    
    # Append the current DataFrame to the main DataFrame
    main_df = pd.concat([main_df, df], ignore_index=True)


export_file = os.path.join(output_folder, 'analysis.csv')
main_df.to_csv(export_file, sep='\t', encoding='utf-8', index=False, header=True)

print("Time needed for running:", datetime.now() - start)