# Script takes segmentation weights, responsible for car, truck and bus classes, from pretrained mask r-cnn (PRETRAINED_PATH)
# and adds them to fine-tuned faster r-cnn weights dictionary (UAV_WEIGHTS_PATH)


import os
import sys
import cv2
import numpy as np
import re
import torch
import argparse
import time
import matplotlib.pyplot as plt
import pickle

# add project root directory to python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_ROOT)

from detectron2.engine.defaults import DefaultPredictor
from detectron2.config import get_cfg

# classes: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

CAR_IDX = 2
TRUCK_IDX = 7
BUS_IDX = 5
PRETRAINED_PATH = '/pretrained/mask_rcnn_R_50_FPN_3x_original/model_final_f10217.pkl'
UAV_WEIGHTS_PATH = '/pretrained/cars_detector_UAV1ep_rand_all_lr5e-3_4im/model_700.pth'


def convert_ndarray_to_tensor(state_dict):
        """
        In-place convert all numpy arrays in the state_dict to torch tensor.
        Args:
            state_dict (dict): a state-dict to be loaded to the model.
                Will be modified.
        """
        # model could be an OrderedDict with _metadata attribute
        # (as returned by Pytorch's state_dict()). We should preserve these
        # properties.
        for k in list(state_dict.keys()):
            v = state_dict[k]
            if not isinstance(v, np.ndarray) and not isinstance(v, torch.Tensor):
                raise ValueError(
                    "Unsupported type found in checkpoint! {}: {}".format(k, type(v))
                )
            if not isinstance(v, torch.Tensor):
                state_dict[k] = torch.from_numpy(v)


uav_weights = torch.load( PROJECT_ROOT + UAV_WEIGHTS_PATH )
with open(PROJECT_ROOT + PRETRAINED_PATH, "rb") as f:
    predictor_weights = pickle.load(f, encoding="latin1")
predictor_weights = predictor_weights['model']
convert_ndarray_to_tensor(predictor_weights)
for k, v in predictor_weights.items():
    if 'roi_heads.mask_head' in k:
        uav_weights[k] = v
    # print(k, v.shape)

car_layer = predictor_weights['roi_heads.mask_head.predictor.weight'][CAR_IDX]
truck_layer = predictor_weights['roi_heads.mask_head.predictor.weight'][TRUCK_IDX]
bus_layer = predictor_weights['roi_heads.mask_head.predictor.weight'][BUS_IDX]
mask_predictor_weight = torch.stack([car_layer, truck_layer, bus_layer], dim=0)

car_bias = predictor_weights['roi_heads.mask_head.predictor.bias'][CAR_IDX]
truck_bias = predictor_weights['roi_heads.mask_head.predictor.bias'][TRUCK_IDX]
bus_bias = predictor_weights['roi_heads.mask_head.predictor.bias'][BUS_IDX]
mask_predictor_bias = torch.tensor([car_bias, truck_bias, bus_bias])

uav_weights['roi_heads.mask_head.predictor.weight'] = mask_predictor_weight
uav_weights['roi_heads.mask_head.predictor.bias'] = mask_predictor_bias

torch.save(uav_weights, './pretrained/mask_rcnn_50_FPN_aerial/model.pth')

