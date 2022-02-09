import sys
import os
import cv2
import torch
import numpy as np

# add project root directory to python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_ROOT)

from detectron2.config import get_cfg
from detectron2.utils.colormap import random_color

from utils.MOT_utils import MOTloader
from networks.association_head import AssociationHead


def associate_colors_to_ids(sequence_objects):

    num_ids = np.max(sequence_objects[:, 1])
    colors = [tuple( int(x) for x in random_color() ) for i in range(num_ids)]
    
    return colors


def visualize_frame(frame_objects, frame, colors, show_ids=False, distances=None, choosen_id=None):
    #sequence_objects: nparray <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>
        
    if distances is not None:
        mindist_idx = np.argmin(distances)

    for i, instance in enumerate(frame_objects):

        id = instance[1]
        color = colors[id - 1]
        topleft_pt = (instance[2], instance[3])
        bottomright_pt = (instance[2] + instance[4], instance[3] + instance[5])
        frame = cv2.rectangle(frame, topleft_pt, bottomright_pt, color, 2)

        text_color = (0, 0, 255) # BGR
        font_scale = 1
        thickness = 1
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL

        if show_ids:

            id_text = '(' + str(id) + ')'

            if distances is not None:
                id_text += ' {}'.format(round(distances[i].item(), 1) )
                if i == mindist_idx:
                    text_color = (0, 255, 0)

            text_pos = (instance[2], instance[3] - 5)
            frame = cv2.putText(frame, id_text, text_pos, font, font_scale, text_color, thickness, cv2.LINE_AA) 
            
        if choosen_id is not None:

            text = 'choosen id: ' + str(choosen_id)
            frame = cv2.putText(frame, text, (20, 20), font, font_scale, text_color, thickness, cv2.LINE_AA) 

    cv2.imshow('frame', frame)


def setup_cfg():

    cfg = get_cfg()
    cfg.merge_from_file(PROJECT_ROOT + '/configs/mask_rcnn_R_101_FPN_3x.yaml')
    # cfg.MODEL.WEIGHTS = PROJECT_ROOT + "/pretrained/mask_rcnn_R_101_FPN_3x_original/model_final_a3ec72.pkl"
    cfg.freeze()

    return cfg


def calculate_distances(anchor_embedding, embeddings):

    distances = [0.0] * embeddings.size()[0]

    for i, embedding in enumerate(embeddings):

        diff = anchor_embedding - embedding
        diff.detach()
        distances[i] = torch.dot(diff, diff)
    
    return distances


sequence_path = PROJECT_ROOT + '/datasets/MOT20/train/MOT20-02'
ROI_SIZE = 10
WEIGHTS_PATH = PROJECT_ROOT + '/pretrained/association_head_N_10ep_R_101_FPN_3x/association_head.pth'

config = setup_cfg()
dataloader = MOTloader(
    config=config, 
    sequence_path=sequence_path, 
    roi_size=ROI_SIZE)
ids_colors = associate_colors_to_ids(dataloader.sequence_objects)
association_head = AssociationHead(roi_size=ROI_SIZE, input_depth=dataloader.roi_generator.get_features_depth() )
association_head.to(torch.device(config.MODEL.DEVICE))
association_head.load_state_dict(torch.load(WEIGHTS_PATH))

for frame_idx in range(dataloader.frames_in_sequence):

    frame_number = frame_idx + 1
    frame_objects = dataloader.objects_from_frame(frame_number)
    frame = dataloader.frame_from_sequence(frame_number)
    ids, rois = dataloader.roi_generator.get_rois_features(frame, frame_objects)
    embeddings = association_head(rois)

    if frame_idx == 0:

        visualize_frame(frame_objects, frame, ids_colors, show_ids=True)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        choosen_id = int(input('Choose object id: ') )
        choosen_idx = np.where(ids.cpu().numpy() == choosen_id)
        choosen_embedding = embeddings[choosen_idx][0]

    else:

        distances = calculate_distances(choosen_embedding, embeddings)
        visualize_frame(frame_objects, frame, ids_colors, show_ids=True, distances=distances, choosen_id=choosen_id)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

cv2.destroyAllWindows()