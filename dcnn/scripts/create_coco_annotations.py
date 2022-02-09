import sys
import os
import cv2
import torch
import argparse
import pafy
import time
import json
import numpy as np
# add project root directory to python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(PROJECT_ROOT)


INPUT_CSV = './test_images/uav/labels.csv'
OUTPUT_JSON = './test_images/uav/annotations.json'
NAME_TO_ID = {
    'car': 0,
    'truck': 1,
    'bus': 2,
    'person': 3
}

images = []
annotations = []
categories = []
img_name_to_id = {}
with open(INPUT_CSV, 'r') as csv_file:
    csv_lines = csv_file.readlines()

for catname, cat_id in NAME_TO_ID.items():

    cat_dict = {
        'id': cat_id,
        'name': catname
    }
    categories.append(cat_dict)

for ann_id, line in enumerate(csv_lines):
    
    line = line.split(',')
    category = NAME_TO_ID[line[0]]
    bbox_xywh = [line[1], line[2], line[3], line[4]]
    bbox_xywh = [int(el) for el in bbox_xywh]
    filename = line[5]
    imgwidth = int(line[6])
    imgheight = int(line[7].strip())

    if filename not in img_name_to_id:
        img_name_to_id[filename] = len(images)
        img_dict = {
            'id': img_name_to_id[filename],
            'file_name': filename,
            'width': imgwidth,
            'height': imgheight
        }
        images.append(img_dict)

    annotation_dict = {
        'iscrowd': 0,
        'image_id': img_name_to_id[filename],
        'category_id': category,
        'bbox': bbox_xywh,
        'id': ann_id,
        'area': bbox_xywh[2] * bbox_xywh[3]
    }
    annotations.append(annotation_dict)


coco_dict = {
    'images': images,
    'annotations': annotations,
    'categories': categories
}

with open(OUTPUT_JSON,'w') as json_file:
    json.dump(coco_dict, json_file, indent = 1)

    