# Collection of scripts and functions;
Usage: calls from root or import the package in a jupyter notebook;
If the package is not found, run `python core/__init__.py`;


## check the core.transforms functions
```python
# custom functions for data augmentation
from core.transforms import *
check_distortion('data/samples/marat.png', SineFold(alpha=4.0))
compare_distortion(SineNoise(axis='x'))
```

An example script for data augmentation is provided;
```bash
python scripts/augment.py
```

## creating your own COCO dataset with utils.dataset

```python
# import all functions
from utils.dataset import *

# generate some data unfolding images
input_dir = 'data/ds25s/test/imgs/'
output_dir = 'data/ds25s/test_aug/imgs/'
folder_fold_ij(input_dir, output_dir, 2, 2, True)
# since we have data from 2 sources (augment.py and fold_ij), rename files
rename_files_in_folder('output_dir', 1)

# do exactly the same for masks
input_dir = 'data/ds25s/test/masks1/'
output_dir = 'data/ds25s/test_aug/masks1/'
folder_fold_ij(input_dir, output_dir, 2, 2, True)
rename_files_in_folder(output_dir, 1)

# if images and masks are correct, proceed with the json generation
data_dir = 'data/ds25s/test_aug/'
masks1_dir = data_dir+'masks1'
output_json = data_dir+'masks.json'
process_directory(masks1_dir, output_json)

# generate the final, clean json
output_path = data_dir+'clean.json'
clean_invalid_segmentations(output_json, output_path)

# get values to optimize training
target_folder = data_dir+'imgs'
mean, std = get_folder_values(target_folder)
```


# Extra
## custom scripts (non-*argparse*)
```bash
# An example script for data augmentation:
python scripts/augment.py
# A script to check the number of parameters in a given model
python scripts/check_param.py
```