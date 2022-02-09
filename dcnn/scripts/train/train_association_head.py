import os
import sys
import cv2
import numpy as np
import torch
import torch.optim as optim
import re
import argparse
import time

np.set_printoptions(threshold=sys.maxsize)
# add project root directory to python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_ROOT)

from detectron2.config import get_cfg
from online_triplet_loss.losses import batch_hard_triplet_loss

from utils.MOT_utils import MOTloader, MOTSloader
from engines.roi_features_generator import RoiFeaturesGenerator
from networks.association_head import AssociationHead

NETWORK = 'R_101_FPN_3x'

def get_parser():
    parser = argparse.ArgumentParser(description="Association head training")
    parser.add_argument('--checkpoint', help='Path to pretrained association_head.pth checkpoint')
    return parser


def setup_cfg():

    cfg = get_cfg()
    cfg.merge_from_file(PROJECT_ROOT + '/configs/mask_rcnn_' + NETWORK + '.yaml')
    cfg.MODEL.WEIGHTS = './pretrained/R_101_FPN_UAV_SEGM_bestAP.pth'
    cfg.freeze()

    return cfg


args = get_parser().parse_args()
FRAMES_IN_BATCH = 6
NUM_EPOCH = 10
ROI_SIZE = 10
LEARNING_RATE = 0.01
MOMENTUM = 0.9

# sequence_path = PROJECT_ROOT + '/datasets/MOT20/train/MOT20-01'
dataset_path = PROJECT_ROOT + '/datasets/data_tracking_image_2'
seqmap_path = PROJECT_ROOT + '/mots_tools/mots_eval/train.seqmap'
#config for backbone
config = setup_cfg()

# dataloader = MOTloader(
#     config=config, 
#     sequence_path=sequence_path, 
#     frames_in_batch=FRAMES_IN_BATCH, 
#     roi_size=ROI_SIZE)
dataloader = MOTSloader(
    config = config,
    dataset_path = dataset_path,
    seqmap_path = seqmap_path,
    frames_in_batch = FRAMES_IN_BATCH,
    roi_size = ROI_SIZE
)


print('Dataset loaded with {} batches and {} sequences'.format(dataloader.num_of_batches, len(dataloader.seqmap_names)))
print('Training for', NUM_EPOCH, 'epochs')
training_start_time = time.time()
# print('number of frames with objects per sequence:')
# for k in dataloader.frames_with_objects_per_seq.keys():
#     print(k, ':', len(dataloader.frames_with_objects_per_seq[k]))

if args.checkpoint:
    CHECKPOINT_NAME = 'association_head_UAV' + 'CHECKPOINT_' + NETWORK
else:
    CHECKPOINT_NAME = 'association_head_UAV_' + 'roi' + str(ROI_SIZE) + '_' + str(NUM_EPOCH) + 'ep_' + NETWORK
PATH = PROJECT_ROOT + '/pretrained/' + CHECKPOINT_NAME
print('Checkpoint will be saved as', PATH)
if not os.path.isdir(PATH):
    os.mkdir(PATH)

association_head = AssociationHead(roi_size=ROI_SIZE, input_depth=dataloader.roi_generator.get_features_depth() )
if args.checkpoint:
    association_head.load_state_dict(torch.load(args.checkpoint))
    print('checkpoint {} loaded'.format(args.checkpoint))
association_head.to(torch.device(config.MODEL.DEVICE))
optimizer = optim.SGD(association_head.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
avglosses_per_epoch = []


for epoch in range(NUM_EPOCH):

    print('EPOCH:', epoch)
    epoch_loss = 0
    for sequence_idx in range(dataloader.num_of_sequences):
        print('Sequence:', dataloader.seqmap_names[sequence_idx])
        sequence_loss = 0
        for batch_idx in range(dataloader.batches_per_sequence[sequence_idx]):

            ids, rois = dataloader.get_training_batch(sequence_idx, batch_idx)
            optimizer.zero_grad()

            # #DEBUG
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # sys.exit()

            embeddings = association_head(rois)
            loss = batch_hard_triplet_loss(ids, embeddings, margin=0.2, device='cuda:0')
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            sequence_loss += loss.item()
            # if batch_idx % 10 == 0:
            #     print('epoch: {}, batch_idx: {}, batch_loss: {}'.format(epoch, batch_idx, loss.item() ) )
        print('\taverage loss in sequence:', sequence_loss/dataloader.batches_per_sequence[sequence_idx])
    
    print('epoch {} finished, average loss: {}'.format(epoch, epoch_loss/dataloader.num_of_batches) )
    avglosses_per_epoch.append(epoch_loss/dataloader.num_of_batches)
    torch.save(association_head.state_dict(), PATH + '/association_head_EP{}.pth'.format(epoch))

print('Training finished')
torch.save(association_head.state_dict(), PATH + '/association_head.pth')
training_end_time = time.time()
total_time = training_end_time - training_start_time
total_time = str(total_time//3600) + 'h' + str((total_time%3600)/60) + 'm'
with open(PATH + '/train_info.txt', 'w+') as file:

    file_string = 'FRAMES_IN_BATCH: ' + str(FRAMES_IN_BATCH) + '\n'
    file_string += 'NUM_EPOCH: ' + str(NUM_EPOCH) + '\n'
    file_string += 'ROI_SIZE: ' + str(ROI_SIZE) + '\n'
    file_string += 'LEARNING_RATE: ' + str(LEARNING_RATE) + '\n'
    file_string += 'MOMENTUM: ' + str(MOMENTUM) + '\n'
    file_string += 'training time: ' + total_time + '\n'
    for loss in avglosses_per_epoch:
        file_string += str(loss) + ','
    file.write(file_string)