# creme  
## Crystal defects Recognition in Electron Microscopy data Ensembles  

### Overview  
This package enables automated recognition of crystal defects in electron microscopy datasets.  

### Author  
[Camilo A. F. Salvador](https://github.com/camilofs)

### Contributors  
Thomas Bylik, Mihai-Cosmin Marinica.

### Reference Paper  
**High-throughput analysis of dislocation loops in irradiated metals using Mask R-CNN**  
Advances in transmission electron microscopy under extreme conditions have enabled in situ experiments to capture vast amounts of data on defect evolution. On the other hand, computer vision models such as Mask R-CNN have become popular in the last few years, enabling fast and accurate segmentation of images of different natures. In the present work, we propose a workflow to label, segment, and analyze irradiation-induced defects in TEM images using Mask R-CNN. The work focuses on interpreting bright-field (BF) videos recorded during the irradiation of three different metallic materials. After establishing a baseline dataset based on austenitic stainless steel 316L, we tested small and large models as the backbone of Mask R-CNN and different hyperparameters for training them. Our best model predicts the areal density of defects in 316L with an accuracy of 83.6 \%. We tested the generalization limits of the trained model to ensure accurate estimations of key physical metrics, including the foreground fraction occupied by defects, the number of detected particles, and their relative sizes — all of which exhibit relative errors below 5\%. At last, the model helps interpret videos concerning two similar irradiation experiments: one with the 16Cr-37Fe-13Mn-34Ni (at. \%) alloy, and another with pure Cr. The model's segmentation clearly captures the different nature of defect evolution between different materials, as expected. Moreover, the proposed workflow not only enables consistent, real-time analysis of small defect loops during in situ TEM experiments but also generates the quantitative data needed to refine mesoscale models.

### How to cite
If you use this package, please cite:  
```bibtex
[BibTeX to be included]
```

### Installation
```bash
pip install -r requirements.txt
```

### Usage
To reproduce data from the reference paper, use the [Jupyter notebook](examples/video3.ipynb). For an independent project, you may call scripts from the terminal as shown in [utils](utils/README.md)

### License
MIT
*This work has been carried out within the  Cross-disciplinary initiative for digital science of the French Alternative Energies and Atomic Energy Commission (CEA), 2025.*

