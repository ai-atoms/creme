'''
python3
creme core.models module
models are based on detectron2
'''
import copy
import datetime
import logging
import os
import time
from random import randint

import numpy as np
import torch
import torchvision.models as models

# Detectron2 imports
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import (
    DatasetCatalog,
    DatasetMapper,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
    detection_utils as utils,
)
from detectron2.data.datasets import register_coco_instances

import detectron2.data.transforms as T
from detectron2.engine import DefaultTrainer
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import COCOEvaluator, inference_context
from detectron2.modeling import BACKBONE_REGISTRY, Backbone
from detectron2.modeling.backbone.fpn import build_resnet_fpn_backbone
from detectron2.modeling.backbone.resnet import ResNet

import detectron2.utils.comm as comm
from detectron2.utils.logger import log_every_n_seconds
from detectron2.utils.visualizer import ColorMode, Visualizer


def get_train_cfg_101d(job_name):
    '''ResNeXt-101_32x8d with optimal hyperparameters'''
    # dataset
    register_coco_instances('train', {}, 'data/ds25d/train_aug/clean.json', 'data/ds25d/train_aug/imgs')
    register_coco_instances('val', {}, 'data/ds25d/test_aug/clean.json', 'data/ds25d/test_aug/imgs')

    # backbone model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.TEST = ("val",)
    cfg.MODEL.PIXEL_MEAN = [149.6, 149.6, 149.6]
    cfg.MODEL.PIXEL_STD = [35.6, 35.6, 35.6]
    cfg.OUTPUT_DIR = str(f'output/{job_name}')
    
    # using pre-trained weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    # cfg.MODEL.WEIGHTS = os.path.join('models', "model_final_2d9806.pkl") # used in case of retraining
    
    cfg.MODEL.RESNETS.DEPTH = 101
    cfg.MODEL.RESNETS.NUM_GROUPS = 32
    cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 256

    # core hyperparameters
    cfg.SOLVER.IMS_PER_BATCH = 1 # increase if you have more mem. avail.
    cfg.SOLVER.BASE_LR = 0.001  # 0.0005 seems a good start
    cfg.SOLVER.MAX_ITER = 40000  
    cfg.SOLVER.CHECKPOINT_PERIOD = 4000
    cfg.TEST.EVAL_PERIOD = 400
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 # select smaller if faster training is needed
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 
    cfg.TEST.DETECTIONS_PER_IMAGE = 512

    # additional hyperparam
    cfg.MODEL.RPN.IN_FEATURES = ['p2', 'p2', 'p3', 'p4', 'p5', 'p6']
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8], [16], [32], [64], [128], [256]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]]
    cfg.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]
    cfg.MODEL.RPN.POSITIVE_FRACTION = 0.50
    cfg.MODEL.RPN.NMS_THRESH = 0.7
    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 4000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 6000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 2000
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ['p2', 'p3', 'p4', 'p5']
    cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25
    cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7
    cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.SOLVER.STEPS = ()

    # resource allocation
    cfg.DATALOADER.NUM_WORKERS = 4
    cuda_avail = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cfg.MODEL.DEVICE = cuda_avail
    
    return cfg


def get_train_cfg_r50a(job_name):
    '''ResNet-50 with optimal FPN hyperparameters'''
    # dataset
    register_coco_instances('train', {}, 'data/ds25d/train_aug/clean.json', 'data/ds25d/train_aug/imgs')
    register_coco_instances('val', {}, 'data/ds25d/test_aug/clean.json', 'data/ds25d/test_aug/imgs')

    # backbone model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.TEST = ("val",)
    cfg.MODEL.PIXEL_MEAN = [149.6, 149.6, 149.6]
    cfg.MODEL.PIXEL_STD = [35.6, 35.6, 35.6]
    cfg.OUTPUT_DIR = str(f'output/{job_name}')
    
    # update the number of classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Change from 80 to 1

    # using pre-trained weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    # cfg.MODEL.WEIGHTS = os.path.join('models', "model_final_f10217.pkl") # local pretrained backbone
        
    # core hyperparameters
    cfg.SOLVER.IMS_PER_BATCH = 8 # increase if you have more mem. avail.
    cfg.SOLVER.BASE_LR = 0.001  # 0.0005 seems a good start
    cfg.SOLVER.MAX_ITER = 20000  
    cfg.SOLVER.CHECKPOINT_PERIOD = 2000
    cfg.TEST.EVAL_PERIOD = 200
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 # select smaller if faster training is needed
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 
    cfg.TEST.DETECTIONS_PER_IMAGE = 512

    # additional hyperparam
    cfg.MODEL.RPN.IN_FEATURES = ['p2', 'p2', 'p3', 'p4', 'p5', 'p6']
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8], [16], [32], [64], [128], [256]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]]
    cfg.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]
    cfg.MODEL.RPN.POSITIVE_FRACTION = 0.50
    cfg.MODEL.RPN.NMS_THRESH = 0.7
    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 4000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 6000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 2000
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ['p2', 'p3', 'p4', 'p5']
    cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25
    cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7
    cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.SOLVER.STEPS = ()

    # resource allocation
    cfg.DATALOADER.NUM_WORKERS = 4
    cuda_avail = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cfg.MODEL.DEVICE = cuda_avail
    
    return cfg


@BACKBONE_REGISTRY.register() # Register ResNet-18 with FPN
def build_resnet18_fpn_backbone(cfg, input_shape):
    '''
    Create a ResNet-18 FPN backbone with pre-trained weights.
    Requires a valid *.pth file
    '''
    cfg.defrost()  # Allow modification
    cfg.MODEL.RESNETS.DEPTH = 18  # Set ResNet depth to 18
    cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 64  # Adjust output channels
    cfg.MODEL.WEIGHTS = ""  # This will be overridden
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    cfg.freeze()  # Prevent further modification

    backbone = build_resnet_fpn_backbone(cfg, input_shape)

    # Load pretrained weights locally
    resnet18_pretrained = models.resnet18()
    resnet18_pretrained.load_state_dict(torch.load('models/resnet18.pth', map_location="cpu"))
    pretrained_dict = resnet18_pretrained.state_dict()

    # Convert keys to Detectron2 format
    detectron2_dict = {}
    for k, v in pretrained_dict.items():
        if "fc" in k:  # Ignore fully connected layer
            continue
        detectron2_dict["backbone.bottom_up." + k] = v  # Match Detectron2 key format

    # Load weights into the model
    backbone.load_state_dict(detectron2_dict, strict=False)

    return backbone


def get_train_cfg_r18a(job_name):
    '''ResNet-18 with optimal FPN hyperparameters'''   
    # dataset
    register_coco_instances('train', {}, 'data/ds25d/train_aug/clean.json', 'data/ds25d/train_aug/imgs')
    register_coco_instances('val', {}, 'data/ds25d/test_aug/clean.json', 'data/ds25d/test_aug/imgs')

    # backbone model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    # Replace ResNet-50 with ResNet-18
    cfg.MODEL.BACKBONE.NAME = "build_resnet18_fpn_backbone"    
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.TEST = ("val",)
    cfg.MODEL.PIXEL_MEAN = [149.6, 149.6, 149.6]
    cfg.MODEL.PIXEL_STD = [35.6, 35.6, 35.6]
    cfg.OUTPUT_DIR = str(f'output/{job_name}')
    
    # update the number of classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Change from 80 to 1
        
    # core hyperparameters
    cfg.SOLVER.IMS_PER_BATCH = 12 # increase if you have more mem. avail.
    cfg.SOLVER.BASE_LR = 0.001  # 0.0005 seems a good start
    cfg.SOLVER.MAX_ITER = 20000  
    cfg.SOLVER.CHECKPOINT_PERIOD = 2000
    cfg.TEST.EVAL_PERIOD = 200
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 # select smaller if faster training is needed
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 
    cfg.TEST.DETECTIONS_PER_IMAGE = 512

    # additional hyperparam
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16], [32], [64], [128], [256]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.1, 0.2, 0.5, 1.0, 2.0, 5.0]]
    cfg.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]
    cfg.MODEL.RPN.POSITIVE_FRACTION = 0.50
    cfg.MODEL.RPN.NMS_THRESH = 0.7
    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 4000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 6000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 2000
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ['p2', 'p3', 'p4', 'p5']
    cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25
    cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7
    cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.SOLVER.STEPS = ()

    # resource allocation
    cfg.DATALOADER.NUM_WORKERS = 4
    cuda_avail = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cfg.MODEL.DEVICE = cuda_avail
    
    return cfg


class LossEvalHook(HookBase):
    # Code most probably from : https://gist.github.com/ortegatron/c0dad15e49c2b74de8bb09a5615d9f6b
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader

    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on the validation dataset {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()
        return losses

    def _get_loss(self, data):
        # Combined loss calculation
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)


class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        ))
        return hooks