import numpy as np
import cv2
import torch

from pycocotools.mask import encode, decode

from utils.mask_utils import show_mask

# time_frame id class_id img_height img_width rle

# # An example line from a txt file:

# 52 1005 1 375 1242 WSV:2d;1O10000O10000O1O100O100O1O100O1000000000000000O100O102N5K00O1O1N2O110OO2O001O1NTga3

# Which means

# time frame 52
# object id 1005 (meaning class id is 1, i.e. car and instance id is 5)
# class id 1
# image height 375
# image width 1242
# rle WSV:2d;1O10000O10000O1O100O100O1O100O1000000000000000O100O...1O1N 


def file_lines_from_instances(object_instances, frame_num, image_size):

    out_string = ""
    for obj_idx in range(len(object_instances)):
        ob_class = object_instances.pred_classes[obj_idx].item()
        #if person or car
        if(ob_class == 0 or ob_class == 2):
            #change labels to MOTS format (car:1, pedestrian:2)
            if ob_class == 0:
                ob_class = 2
            else:
                ob_class = 1
            ob_id = object_instances.ids[obj_idx]
            img_height = image_size[0]
            img_width = image_size[1]
            ob_mask = object_instances.pred_masks[obj_idx].cpu().numpy()

            # mask = np.uint8(ob_mask)*255
            # cv2.imshow(str(obj_idx) + ' przed', mask)

            ob_rle = encode(np.asfortranarray(ob_mask) )
            print('rle type:', type(ob_rle))

            # dec_mask = decode(ob_rle)
            # cv2.imshow(str(obj_idx) + ' po', dec_mask*255)

            object_id = ob_class * 1000 + ob_id
            line = str(frame_num) + ' ' + str(object_id) + ' ' + str(ob_class) + ' ' + str(img_height) + ' ' + str(img_width) + ' ' + str(ob_rle['counts'])[2:-1] + '\n'
            out_string += line

    return out_string


def result_image_from_objects(object_instances, image_size):

    img = np.zeros(image_size, dtype=np.uint16)
    for obj_idx in range(len(object_instances)):
        ob_class = object_instances.pred_classes[obj_idx].item()
        #if person or car
        if(ob_class == 0 or ob_class == 2):
            #change labels to MOTS format (car:1, pedestrian:2)
            if ob_class == 0:
                ob_class = 2
            else:
                ob_class = 1
            ob_id = object_instances.ids[obj_idx]
            ob_mask = object_instances.pred_masks[obj_idx].cpu().numpy()
            object_id = ob_class * 1000 + ob_id
            # print('object id:', object_id)
            img[ob_mask] = object_id

    img = img.astype(np.uint16)
    return img


def parse_mots_seqmap(path):

    sequence_names = []
    sequence_lengths = []

    with open(path, 'r') as FILE:
        file_lines = FILE.readlines()
        for line in file_lines:
            seqname = line.split(' ')[0].strip()
            #indexes of last frame are given in mots seqmaps (frames are indexed from 0)
            seqlen = int(line.split(' ')[3].strip() ) + 1
            sequence_names.append(seqname)
            sequence_lengths.append(seqlen)

    return sequence_names, sequence_lengths


def crop_overlapping_masks(object_instances):

    if len(object_instances) == 0: return

    masks = object_instances.pred_masks
    scores = object_instances.scores
    objects_dict = object_instances.get_fields()

    for i in range(len(masks)):
        # show_mask(masks[i], 'mask ' + str(i))
        for j in range(len(masks))[i+1:]:
            intersection = masks[i] * masks[j]
            if intersection.any():
                # show_mask(masks[i], 'maska')
                # show_mask(intersection, 'overlap')
                # show_mask(torch.logical_xor(masks[i], intersection), 'przycieta')
                if scores[i] > scores[j]:
                    objects_dict['pred_masks'][j] = torch.logical_xor(masks[j], intersection)
                else:
                    objects_dict['pred_masks'][i] = torch.logical_xor(masks[i], intersection)
                # print(str(i) + ' and ' + str(j) + 'are overlapping')
                # show_mask(intersection, str(i) + str(j) + ' intersection')
    
    # for i in range(len(masks)):
    #     show_mask(masks[i], 'mask ' + str(i))

    # cv2.waitKey(0)



