# Collection of calls;
Usage: calls from root via *argparse* or import the package in a jupyter notebook;
It the package is not found, run 'python core/__init__.py';


## core.transforms module
```python
# explicit import of custom functions for data augmentation
from core.transforms import *
check_distortion('data/samples/marat.png', SineFold(alpha=4.0))
compare_distortion(SineNoise(axis='x'))
```





