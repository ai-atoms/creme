'''
python3 
[creme] training a new model
'''

import logging
import os
import pickle
import shutil
from pathlib import Path

import torch
from detectron2.utils.logger import setup_logger

from core.models import MyTrainer, get_train_cfg_r50a

setup_logger()  # setup might vary in a HPC env

# avoid repeated names
job_name = 'ds25d_r50a_150525'
'''
d2 ResNet50-FPN
Pre-trained weights from source 
(loops only = 1 class)
ds25d dataset
15 May 2025
'''
Path(f'output/{job_name}').mkdir(parents=True, exist_ok=True)

# get model config
cfg = get_train_cfg_r50a(job_name)

# train model
trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# copy the script to the output folder just in case
shutil.copyfile(__file__, cfg.OUTPUT_DIR + __file__.split("/")[-1])