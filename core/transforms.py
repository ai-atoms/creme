'''
python3
creme core.transforms module
custom classes are compatible with torchvision.transforms.v2
'''

from functools import partial, singledispatchmethod
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import tv_tensors
import torchvision.transforms.v2 as transforms


def create_rotated_image_stack(inpt, batch_size=8, max_angle=30):
    # helper function
    # based on pytorch.org/vision/stable/auto_examples/transforms/
    rotater = transforms.RandomRotation(degrees=(0, max_angle))
    if isinstance(inpt, Image.Image):
        inpt = transforms.PILToTensor()(inpt)
    rotated_inpts = [rotater(inpt) for _ in range(batch_size)]
    return torch.stack(rotated_inpts)


def tensor_to_numpy(img, normalize=False):
    # helper function
    img = img.permute(1, 2, 0).numpy()
    if normalize:
        img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1] for visualization
    return img
    

def svd_denoise(array, r):
    # r is the rank;
    if r ==  None:
        return array
    array_copy = array.copy()
    U, S, VT = np.linalg.svd(array_copy, full_matrices=False)
    S = np.diag(S)
    reconstructed = U[:, r:] @ S[r:, r:] @ VT[r:, :]
    #projector = U[:, :r] @ np.transpose(U[:, :r])
    #projected = projector @ array_copy
    return reconstructed #, projected,  projector;

    
class SineNoise(transforms.Transform):
    '''
    A torchvision V2 compatible transform that applies a periodic noise to a tensor
    This class operates without data interpolation (!= than ElasticTransform)
    Adapted from a base version by A. Dezaphie, CEA Saclay, 2023.
    Footer from a custom v2 class adapted from github.com/cj-mills
    alpha_x = 0.2 and alpha_y = 1.6 are good values for 512x512 px images
    TBD: Allocate tensors to a device; replace loops with torch.nn.functional.grid_sample if possible;
    '''
    def __init__(self,
             alpha_x: float = 0.2, # The perturbation amplitude in x
             alpha_y: float = 1.6, # The perturbation amplitude in y
             axis: str = 'x',   # x -> shift will be applied to the x-axis
             ):
        super().__init__()
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y
        self.axis = axis

    def shift(self, r):
        return int(self.alpha_y * np.sin(np.pi * r * self.alpha_x))

    def distort(self, img: torch.Tensor) -> torch.Tensor:
        if img.dim() == 3: 
            # single img of shape (C, H, W)
            img_dist = torch.empty_like(img)
            if self.axis == 'x':
                # apply vertical distortion
                for i in range(img.size(2)):
                    img_dist[:, :, i] = torch.roll(img[:, :, i], shifts=self.shift(i), dims=1)
            else:
                # apply horizontal distortion
                for i in range(img.size(1)):
                    img_dist[:, i, :] = torch.roll(img[:, i, :], shifts=self.shift(i), dims=1)
            return img_dist
        
        elif img.dim() == 4:
            # N imgs (N, C, H, W)
            img_dist = torch.empty_like(img)
            # print (img_dist.shape)
            if self.axis == 'x':
                for n in range(img.size(0)):  # iterate over batch (N)
                    for i in range(img.size(3)):
                        img_dist[n, :, :, i] = torch.roll(img[n, :, :, i], shifts=self.shift(i), dims=1)
            else:
                for n in range(img.size(0)):  # iterate over batch (N)
                    for i in range(img.size(2)):
                        img_dist[n, :, i, :] = torch.roll(img[n, :, i, :], shifts=self.shift(i), dims=1)
            return img_dist
        
        else:
            raise ValueError("Input tensor should have 3 (C, H, W) or 4 (N, C, H, W) dimensions.")

    @singledispatchmethod
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        """Default Behavior: Don't modify the input"""
        return inpt

    @_transform.register(torch.Tensor)
    @_transform.register(tv_tensors.Image)
    def _(self, inpt: Union[torch.Tensor, tv_tensors.Image], params: Dict[str, Any]) -> Any:
        """Apply the 'distort' method to the input tensor"""
        return self.distort(inpt)

    @_transform.register(Image.Image)
    def _(self, inpt: Image.Image, params: Dict[str, Any]) -> Any:
        """Convert the PIL Image to a torch.Tensor to apply the transform"""
        inpt_torch = transforms.PILToTensor()(inpt)
        return transforms.ToPILImage()(self._transform(inpt_torch, params))


class SineFold(transforms.Transform):
    '''
    Similar than SineNoise, but operates at a larger scale,
    thus, it produces a folding effect avoiding local information loss
    This function was optimized for tensors with F.grid_sample
    Values are scaled in case it receives a PIL image
    alpha = 4.0 is a good value for 512x512 px images
    TBD: Implement the (image, mask) case
    '''
    def __init__(self,
             alpha: float = 4.0, # The perturbation amplitude (pct)
             axis: str = 'x',   # x -> shift will be applied to the x-axis
             ):
        super().__init__()
        self.alpha = alpha
        self.axis = axis
    
    def generate_grid(self, H, W, device):
        # Generate a regular grid
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        grid_x = grid_x.float() / (W - 1) * 2 - 1  # Normalize to [-1, 1]
        grid_y = grid_y.float() / (H - 1) * 2 - 1  # Normalize to [-1, 1]

        # Apply sine distortion
        if self.axis == 'x':
            displacement = self.alpha/100 * torch.sin(grid_y * np.pi)
            grid_x += displacement
        else: 
            displacement = self.alpha/100 * torch.sin(grid_x * np.pi)
            grid_y += displacement

        # Stack grids and reshape to (H, W, 2)
        grid = torch.stack((grid_x, grid_y), dim=-1)
        grid = grid.unsqueeze(0)  # Add batch dimension
        return grid

    # -- alternative _distort
    def _distort(self, img: torch.Tensor) -> torch.Tensor:
        reduce = False
        img = img.float() # avoid F.grid_sample exception

        if img.dim() == 3:
            # Single img of shape (C, H, W) -> (N, C, H, W)
            img = img.unsqueeze(0) # adjust dimensions
            reduce = True
        
        N, C, H, W = img.shape
        grid = self.generate_grid(H, W, img.device).repeat(N, 1, 1, 1)
        # img_dist = F.grid_sample(img, grid, align_corners=True)
        img_dist = F.grid_sample(img, grid, mode='nearest', align_corners=False)
        
        if reduce:            
            return img_dist.squeeze(0) # return to original dimensions
        elif img.dim() == 4:
            return img_dist
        else: 
            raise ValueError("Input tensor should have 3 (C, H, W) or 4 (N, C, H, W) dimensions.")


    @singledispatchmethod
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        """Default Behavior: Don't modify the input"""
        return inpt

    @_transform.register(torch.Tensor)
    @_transform.register(tv_tensors.Image)
    def _(self, inpt: Union[torch.Tensor, tv_tensors.Image], params: Dict[str, Any]) -> Any:
        """Apply the 'distort' method to the input tensor"""
        return self._distort(inpt)

    @_transform.register(Image.Image)
    def _(self, inpt: Image.Image, params: Dict[str, Any]) -> Any:
        """Convert the PIL Image to a torch.Tensor to apply the transform"""
        inpt_torch = transforms.PILToTensor()(inpt)
        inpt_torch = inpt_torch / 255 # norm to [0, 1]
        return transforms.ToPILImage()((self._transform(inpt_torch, params))*1)


def compare_distortion(distortion, inset=True):
    # sanity check
    if inset:
        X = Image.open(Path('data/samples') / 'marat_inset.png').convert('L')
    else:
        X = Image.open(Path('data/samples') / 'marat.png').convert('L')
    y = torch.rand((1, 512, 512)) # (C, H, W)

    image = X
    mask = y
    
    # -- example usage (image and mask)
    image_t = distortion(image)
    mask_t = distortion(mask)    

    # plot section
    fig, axs = plt.subplots(2, 2, figsize=(8,8))
    plt.set_cmap('cubehelix')
    axs[0, 0].imshow(image)
    axs[0, 0].set_title('Image')
    axs[0, 1].imshow(image_t)
    axs[0, 1].set_title('Trans. Image')
    axs[1, 0].imshow(tensor_to_numpy(mask))
    axs[1, 0].set_title('Tensor')
    axs[1, 1].imshow(tensor_to_numpy(mask_t))
    axs[1, 1].set_title('Trans. Tensor')
    fig.tight_layout()
    plt.show()    

    # example usage (batch of images)
    batch = create_rotated_image_stack(X, batch_size=4)
    batch_t = distortion(batch)

    fig, axs = plt.subplots(2, 2, figsize=(8,8))
    axs[0, 0].imshow(tensor_to_numpy(batch[0]))
    axs[0, 0].set_title('First image')
    axs[0, 1].imshow(tensor_to_numpy(batch_t[0]))
    axs[0, 1].set_title('First image, trans')
    axs[1, 0].imshow(tensor_to_numpy(batch[-1]))
    axs[1, 0].set_title('Last image')
    axs[1, 1].imshow(tensor_to_numpy(batch_t[-1]))
    axs[1, 1].set_title('Last image, trans')
    fig.tight_layout()
    plt.show()


def check_distortion(image_path, distortion):
    X = Image.open(Path(image_path))
    image_t = distortion(X)
    
    # plot section
    fig, ax = plt.subplots(figsize=(8,8))
    plt.set_cmap('cubehelix')
    ax.imshow(image_t)
    fig.tight_layout()
    plt.show()


# -- Example usage
if __name__ == "__main__":
    print ('[creme] core.transforms module loaded;')
