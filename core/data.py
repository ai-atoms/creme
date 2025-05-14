'''
python3
creme core.data module
custom classes are compatible with torchvision.datasets
'''

from pathlib import Path
from typing import Callable, Optional, Union

from PIL import Image
import torch
from torchvision.datasets import VisionDataset
from torchvision.transforms.functional import to_tensor


class TwinsDataset(VisionDataset):
    def __init__(
        self,
        root_dir:  Union[str, Path],
        mask1_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        # inherit VisionDataset properties
        super().__init__(root_dir, transforms, transform, target_transform)

        self.root_dir  = Path(root_dir)
        self.mask1_dir = Path(mask1_dir)

        # Get list of image file names in the root directory
        self.image_files = sorted(self.root_dir.glob("*"))
        self.mask1_files = sorted(self.mask1_dir.glob("*"))

        assert len(self.image_files) == len(self.mask1_files), \
            "Mismatch in the number of images and masks!"

        # Map filenames without extensions for matching
        self.image_map = {file.stem: file for file in self.image_files}
        self.mask1_map = {file.stem: file for file in self.mask1_files}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Get filenames
        image_file = self.image_files[idx]
        stem = image_file.stem  # Extract the base name without extension
        mask1_file = self.mask1_map[stem]

        # Load images
        image = Image.open(image_file).convert("L")
        mask1 = Image.open(mask1_file).convert("L")

        # Convert to tensor
        image = to_tensor(image)
        mask1 = to_tensor(mask1)

        # Move to GPU if available
        if torch.cuda.is_available():
            device = torch.device('cuda')
            image = image.to(device)
            mask1 = mask1.to(device)

        # Apply the same transform using a fixed seed
        if self.transforms:
            state = torch.get_rng_state() # get the current random state
            image = self.transforms(image)
            torch.set_rng_state(state) # ensure the same transformation for the mask1
            mask1 = self.transforms(mask1)

        else:
            logging.info('Loading data without any transformation.')

        # Apply target-specific transforms if provided
        if self.target_transform:
            print ('Not implemented.')
            # mask1 = self.target_transform(mask1)

        return image, mask1

# -- Example usage
if __name__ == "__main__":
    print ('[creme] core.data module loaded;')