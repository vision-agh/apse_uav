import sys
import os
import cv2
import torch
import argparse
import time
import json
import math
import numpy as np
# add project root directory to python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_ROOT)

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor

from engines.rcnn_tracker import RcnnTracker
from utils.track_visualizer import TrackVisualizer
from utils.mots_evaluation import file_lines_from_instances, parse_mots_seqmap, crop_overlapping_masks, result_image_from_objects
from utils.mask_utils import get_mask_centroid, compute_closest_point


TEST_NAME = '101_AUGMENTED'
SEQUENCES_DIR = './test_images'
TO_TEST = ['DJI_0167_60kmh.MP4', 'DJI_0169_70kmh.MP4']
FORD_IDS = [3, 4]   # per video
START_FROM_FRAME = [180, 480]

CONFIDENCE_THRESHOLD = 0.5
CLASSES_NAMES = ['car', 'truck', 'bus', 'person']
NUM_CLASSES = len(CLASSES_NAMES)
GENERATE_LOG = False
WRITE_IMAGES = True
RESIZE_IMAGE = (1240, 720)  # or None
ASSOCIATION_WEIGHTS = './pretrained/101_AUGMENTED/association_head_EP9.pth'
MODEL_WEIGHTS = './pretrained/101_AUGMENTED/R_101_FPN_UAV_SEGM_bestAP.pth'
CAM_PARAMS_FILE = './test_images/cam_params_4k_70.json'
# with open(CAM_PARAMS_FILE, 'r') as file:
#     CAM_PARAMS = json.load(file)


def setup_cfg():

    cfg = get_cfg()
    cfg.merge_from_file(PROJECT_ROOT + '/configs/mask_rcnn_R_101_FPN_3x.yaml')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.WEIGHTS = MODEL_WEIGHTS
    cfg.MODEL.MASK_ON = True
    cfg.freeze()

    return cfg


def preprocess_img(frame):

    ret = CAM_PARAMS["ret"]
    mtx = np.array(CAM_PARAMS["mtx"])
    dist = np.array(CAM_PARAMS["dist"])

    frame = cv2.undistort(frame, mtx, dist, None, None)
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    gamma = 2
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i/255.0, gamma) * 255.0, 0, 255)
    lab[...,0] = cv2.LUT(lab[...,0], lookUpTable)
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return frame


def visualize_tracks(frame, objects, video_visualizer):

    objects = objects.to(torch.device('cpu'))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    vis_frame = video_visualizer.draw_instance_predictions(frame, objects)

    # Converts Matplotlib RGB format to OpenCV BGR format
    vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
    return vis_frame


def get_images_from_dir(path):

    extensions = ['jpg', 'png', 'bmp']
    contents = os.listdir(path)
    contents.sort()
    image_contents = [imgpath for imgpath in contents if imgpath.split('.')[-1] in extensions]

    return image_contents


def generate_log_lines(objects, frame_idx):

    if len(objects) == 0:
        return ''
    else:
        result = []
        centroids = [get_mask_centroid(mask) for mask in objects.pred_masks]
        if FORD_ID in objects.ids:
            ford_idx = [idx for idx, obj_id in enumerate(objects.ids) if obj_id == FORD_ID][0]
            ford_centroid = centroids[ford_idx]
            distances_to_ford = [distance(ford_centroid, centroid) for centroid in centroids]
        else:
            distances_to_ford = ['nan'] * len(objects)
        
        # print('distances:', len(objects), distances_to_ford)
        object_lines = ['{},{},{},{},{},{}'.format(frame_idx, obj_id, CLASSES_NAMES[obj_class], int(centroid[0]), int(centroid[1]), dist)
                        for obj_id, obj_class, centroid, dist
                        in zip(objects.ids, objects.pred_classes, centroids, distances_to_ford)]

        return ('\n').join(object_lines)


def generate_log_oneline(objects, FORD_ID, frame_idx):

    if len(objects) == 0:
        return '', 0
    else:
        result = []
        centroids = [get_mask_centroid(mask) for mask in objects.pred_masks]
        if FORD_ID in objects.ids:
            ford_idx = [idx for idx, obj_id in enumerate(objects.ids) if obj_id == FORD_ID][0]
            ford_centroid = centroids[ford_idx]
            closest_points_to_ford = [compute_closest_point(mask, ford_centroid) for mask in objects.pred_masks]
        else:
            closest_points_to_ford = [('nan', 'nan')] * len(objects)

        result.append(str(frame_idx))
        highest_id = max(objects.ids)
        for ob_id in range(1, highest_id+1):

            if ob_id in objects.ids:
                ob_idx = np.where(np.array(objects.ids) == ob_id)[0][0]
                result += [ str(centroids[ob_idx][0]), str(centroids[ob_idx][1]), str(closest_points_to_ford[ob_idx][0]), str(closest_points_to_ford[ob_idx][1]) ]
            else:
                result += [''] * 4

    return (',').join(result), highest_id


def distance(a, b):

    xdiff = a[0] - b[0]
    ydiff = a[1] - b[1]

    return math.sqrt(xdiff**2 + ydiff**2)


config = setup_cfg()
metadata = MetadataCatalog.get("coco_2017_val")
video_visualizer = TrackVisualizer({'thing_classes': CLASSES_NAMES})

for FORD_ID, vidname, start_frame in zip(FORD_IDS, TO_TEST, START_FROM_FRAME):

    print('Testing video:', vidname)
    SEQUENCE_PATH = os.path.join(SEQUENCES_DIR, vidname)
    LOG_FILE = os.path.join(PROJECT_ROOT, 'output', TEST_NAME, os.path.basename(SEQUENCE_PATH), 'log_file.csv')
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output', TEST_NAME, os.path.basename(SEQUENCE_PATH), 'images')

    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    log_lines = []
    max_obj_id = 0
    if os.path.exists(SEQUENCE_PATH):

        if '.MP4' in SEQUENCE_PATH or '.mp4' in SEQUENCE_PATH:

            video = cv2.VideoCapture(SEQUENCE_PATH)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            image_size = (height, width)
            print(image_size)
            tracker = RcnnTracker(config, image_size, metadata=metadata, weights=ASSOCIATION_WEIGHTS)
            frame_idx = 0
            while(video.isOpened()):

                print('frame: {}/{} xdd'.format(frame_idx+1, frame_count), end='\r')
                if frame_idx < start_frame:
                    video.read()
                    frame_idx += 1
                    continue

                ret, frame = video.read()
                if not ret:
                    break
                # frame = preprocess_img(frame)

                torch.cuda.synchronize()
                start_time = time.time()
                current_objects = tracker.next_frame(frame)
                torch.cuda.synchronize()
                end_time = time.time()
                fps = 1/(end_time - start_time)
                # print('fps:', fps, end='\r')

                if GENERATE_LOG:
                    line, highest_id = generate_log_oneline(current_objects, FORD_ID, frame_idx)
                    # print(lines)
                    log_lines.append(line)
                    if highest_id > max_obj_id:
                        max_obj_id = highest_id
                
                visualized_result = visualize_tracks(frame, current_objects, video_visualizer)
                if WRITE_IMAGES:
                    if RESIZE_IMAGE:
                        visualized_result = cv2.resize(visualized_result, RESIZE_IMAGE)
                    cv2.imwrite(os.path.join(OUTPUT_DIR, 'image_{:04d}'.format(tracker.frame_count) + '.png'), visualized_result)
                
                # cv2.imshow('tracking', visualized_result)
                # key = cv2.waitKey(1)
                # if key == ord('q'):
                #     break
                # elif key == ord('l'):
                #     cv2.imwrite('frame' + str(tracker.frame_count) +'.jpg', visualized_result)

                frame_idx += 1

        if GENERATE_LOG:
            with open(LOG_FILE, 'w') as log_file:
                header = 'frame'
                for id in range(1, max_obj_id+1):
                    header += ',id_{} cent_x,id_{} cent_y,id_{} clos_x,id_{} clos_y'.format(id, id, id, id)
                header += '\n'
                # log_file.write('frame,object_id,object_class,centroid_x[px],centroid_y[px],distance[px]\n')
                
                log_file.write('Ford id: {}\n'.format(FORD_ID) )
                log_file.write(header)
                log_file.write(('\n').join(log_lines))

        cv2.destroyAllWindows()

    else:
        print('path not found: ', SEQUENCE_PATH)

