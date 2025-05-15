'''
python3
a simplified data augmentation script
'''
import os
import sys
import time
from pathlib import Path
sys.path.append('.') # enables calls from root dir

import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import v2

# [creme] imports
import core
from core.data import TwinsDataset
from core.transforms import tensor_to_numpy, SineFold

# presets
torch.manual_seed(1690)

# define paths 
dir_root  = Path('data/ds25s/test/imgs')
dir_mask1 = Path('data/ds25s/test/masks1')
output_path = "data/ds25s/test_aug/"
out_root  = Path(f'{output_path}imgs')
out_mask1 = Path(f'{output_path}masks1')

# check if outpaths exist
os.makedirs(out_root, exist_ok=True)
os.makedirs(out_mask1, exist_ok=True)

# select transforms
transforms = v2.Compose([
    v2.RandomApply([v2.RandomRotation((0, 45))], p=0.25),
    v2.RandomApply([SineFold(alpha=8.0)], p=0.25),
    v2.RandomCrop(size=(512, 512)), # i.e. i=8, j=8;
    v2.RandomHorizontalFlip(p=0.25), 
    v2.RandomVerticalFlip(p=0.25),
    v2.RandomApply([v2.ColorJitter(brightness=0.2, contrast=0.2)], p=0.25),
    v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=0.2)], p=0.25)
])

# create a custom dataset
t_dataset = TwinsDataset(dir_root, dir_mask1, transforms=transforms)

# instance of a data loader (data will be generated every __getitem__ call)
t_dataloader = DataLoader(t_dataset, batch_size=1, shuffle=True)

# run for 16 images
estart = 5
epochs = 16

for epoch in range(estart, epochs + estart):
    for batch in t_dataloader:
        image, mask1 = batch[0][0], batch[1][0]
        
        # -- export image
        image = image.to('cpu')
        image = tensor_to_numpy(image, normalize=True)
        cv2.imwrite(f'{out_root}/{epoch}.png', image*255)

        # -- export mask1
        mask1 = mask1.to('cpu')
        mask1 = tensor_to_numpy(mask1)
        cv2.imwrite(f'{out_mask1}/{epoch}.png', mask1*255)
        break
