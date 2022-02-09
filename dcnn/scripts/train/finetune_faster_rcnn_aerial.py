import os
import sys
import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
import re
import argparse
import time
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)
# add project root directory to python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_ROOT)

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.solver import build_optimizer
from detectron2.utils.events import EventStorage
from detectron2.checkpoint import DetectionCheckpointer

from utils.UAV_utils import UAVDataloader


NUM_EPOCHS = 1
ITERATIONS = 1000
DATASET_DIR = './datasets/UAV-benchmark-M'
OUTPUT_DIR = './pretrained'
OUTPUT_CHECKPOINT_NAME = 'cars_detector_UAV' + str(NUM_EPOCHS) + 'ep_rand_all_lr5e-3_4im_shadow'
LEARNING_RATE = 0.005
MOMENTUM = 0.9
NUM_CLASSES = 3
CHECKPOINT_STEP = 50
IMGS_PER_BATCH = 3
FINETUNE_RPN_REGRESSION = False


def setup_cfg():

    cfg = get_cfg()
    cfg.merge_from_file(PROJECT_ROOT + '/configs/mask_rcnn_R_50_FPN_3x.yaml')
    # cfg.MODEL.WEIGHTS = "./pretrained/mask_rcnn_R_50_FPN_3x_original/model_700.pth"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.MASK_ON = False
    cfg.freeze()

    return cfg


def draw_bboxes(img, boxes):

    for box in boxes:
        start_point = (int(box[0]), int(box[1]))
        end_point = (int(box[2]), int(box[3]))
        result = cv2.rectangle(img, start_point, end_point, (0, 0, 255), 1)
    return result


if not os.path.isdir(os.path.join(OUTPUT_DIR, OUTPUT_CHECKPOINT_NAME) ):
    os.mkdir(os.path.join(OUTPUT_DIR, OUTPUT_CHECKPOINT_NAME))
cfg = setup_cfg()
model = build_model(cfg)
if FINETUNE_RPN_REGRESSION:
    print('Finetuning RPN and bbox regression in box_head')
    parameters = [p for name, p in model.named_parameters() if 'proposal_generator' in name or 'roi_heads.box_predictor.bbox_pred' in name]
else:
    parameters = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(parameters, lr=LEARNING_RATE, momentum=MOMENTUM)
dataloader = UAVDataloader(cfg, DATASET_DIR, imgs_per_batch = IMGS_PER_BATCH, randomize = True)
checkpointer = DetectionCheckpointer(model)
checkpointer.load(cfg.MODEL.WEIGHTS)
model.train()
print('device:', model.device)
avg_cls_loss_per_epoch = []
training_start_time = time.time()
print('Loaded', dataloader.num_batches, 'images.')
print('Training sequences:')
for seq in dataloader.sequences:
    print(seq)


with open(os.path.join(OUTPUT_DIR, OUTPUT_CHECKPOINT_NAME, 'train_info.txt'), 'w+') as file:

    file_string = 'NUM_EPOCH: ' + str(NUM_EPOCHS) + '\n'
    file_string += 'LEARNING_RATE: ' + str(LEARNING_RATE) + '\n'
    file_string += 'MOMENTUM: ' + str(MOMENTUM) + '\n'
    file.write(file_string)


with EventStorage(0) as storage:

    dataloader_iter = iter(dataloader)
    for iteration in range(ITERATIONS):

        data = next(dataloader_iter)
        for el in data:
            print(el['file_name'])
        loss_dict = model(data)
        losses = sum(loss_dict.values())
        print('{:d}/{:d}:\t{:f}'.format(iteration, ITERATIONS, loss_dict['loss_cls'].item()))
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        if iteration % CHECKPOINT_STEP == 0:
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, OUTPUT_CHECKPOINT_NAME, 'model_' + str(iteration) + '.pth') )
            print('Checkpoint saved under', os.path.join(OUTPUT_DIR, OUTPUT_CHECKPOINT_NAME, 'model_' + str(iteration) + '.pth'))


# with EventStorage(0) as storage:

#     for epoch in range(NUM_EPOCHS):
#         print('EPOCH', epoch)
#         epochloss = 0
#         for data in dataloader:
#             for el in data:
#                 print(el['file_name'])
#                 # print(el['instances'].gt_classes)
#             loss_dict = model(data)  
#             losses = sum(loss_dict.values())
#             epochloss += loss_dict['loss_cls'].item()
#             print(loss_dict['loss_cls'].item())
#             optimizer.zero_grad()
#             losses.backward()
#             optimizer.step()
#         avg_loss = epochloss / dataloader.num_batches
#         print('\taverage loss:', avg_loss)
#         avg_cls_loss_per_epoch.append(avg_loss)
#         # plt.plot(avg_cls_loss_per_epoch)
#         # plt.show()


print('Training finished after', NUM_EPOCHS, 'epochs.')
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, OUTPUT_CHECKPOINT_NAME, 'model_final.pth') )
print('Checkpoint saved under', os.path.join(OUTPUT_DIR, OUTPUT_CHECKPOINT_NAME))

training_end_time = time.time()
total_time = training_end_time - training_start_time
total_time = '{:d}h {:d}min'.format(int(total_time//3600), int((total_time%3600)/60))

with open(os.path.join(OUTPUT_DIR, OUTPUT_CHECKPOINT_NAME, 'train_info.txt'), 'a+') as file:

    file_string += 'training time: ' + total_time + '\n'
    for loss in avg_cls_loss_per_epoch:
        file_string += str(loss) + ','
    file.write(file_string)