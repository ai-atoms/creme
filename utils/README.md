# Collection of scripts and functions;
Usage: calls from root or import the package in a jupyter notebook;
If the package is not found, run `python core/__init__.py`;


## Check the core.transforms functions
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

## Create your own COCO dataset with utils.dataset

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

## Setup to extract frames from videos with utils.preproc
To avoid conflicts with detectron2 (which works best in python 3.9), we recommend a python 3.12+ envinronment to run the preproc functions. In this new environment, you may use as reference:

```bash
pip install -r requirements_hspy.txt
```

Minimal examples on how to use the main functions from this module are provided below:

## Gatan(R) nested file tree
### integrate_frames_gatan
```python
from utils.preproc import *
target_folder = 'data/media/video3/Hour_00/Minute_00/Second_42'
output_file = 'output/00m42s.tif'
integrate_frames_gatan(target_folder, output_file)
```

### process_hour_gatan
```python
target_folder = 'data/media/video3/Hour_00'
output_folder = 'data/media/video3/base/'
process_hour_gatan(target_folder, output_folder, clahe=True)
```

### export_accelerated_video
```python
root_path = 'data/media/video3/Hour_00'
fps = 14
output_path = 'output/output_video.mp4' 
export_accelerated_video(root_path, fps, output_path)
```

## MP4 files
### extract_frames_mp4
```python
input_path = 'output/output_video.mp4'
output_path = 'output/'
extract_frames_mp4(input_path, output_path)
```

### crop_video_time
```python
input_video = 'output/output_video.mp4'
output_video = 'output/cropped_video.mp4'
time_limit = 73
crop_video_time(input_video, output_video, time_limit)
```

### crop_center_video
```python
input_video = 'output/cropped_video.mp4'
output_video = 'output/cropped_centered_video.mp4'
crop_center_video(input_video, output_video, 512)
```


## Catalog of custom scripts (non-*argparse*)
```bash
# An example script for data augmentation:
python scripts/augment.py
# A script to check the number of parameters in a given model
python scripts/check_param.py
```