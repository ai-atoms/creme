'''
python3
a simple script to check the number of parameters in a given model
'''
import os
import sys
import time
from pathlib import Path
sys.path.append('.') # enables calls from root dir

# Build model from config
import torch
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

from core.models import get_train_cfg_r18a

# get cfg file
job_name = 'ds25d_r18a_210325'
Path(f'output/{job_name}').mkdir(parents=True, exist_ok=True)
cfg = get_train_cfg_r18a(job_name)

# build model
model = build_model(cfg)
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)  # Load pretrained weights

# count trainable parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Trainable Parameters: {total_params / 1e6:.2f}M")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Backbone Params: {count_parameters(model.backbone) / 1e6:.2f}M")
print(f"RPN Params: {count_parameters(model.proposal_generator) / 1e6:.2f}M")
print(f"ROI Head Params: {count_parameters(model.roi_heads) / 1e6:.2f}M")


