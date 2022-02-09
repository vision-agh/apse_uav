from os.path import join
import os

import numpy as np
from PIL import Image

from detectron2.structures.boxes import Boxes, BoxMode


IGNORED_VISDRONE_CATEGORIES = [0, 3, 7, 8, 10, 11]
VISDRONE_CATEGORY_IDS_TO_UAV = {
    1: 3, #visdrone pedestrian
    2: 3, #visdrone people
    4: 0, #visdrone car
    5: 0, #visdrone van
    6: 1, #visdrone truck
    9: 2 #visdrone bus
}


def get_images_from_dir(path):

    extensions = ['jpg', 'png', 'bmp']
    contents = os.listdir(path)
    contents.sort()
    image_contents = [imgpath for imgpath in contents if imgpath.split('.')[-1] in extensions and 'Annotated' not in imgpath]

    return image_contents


def read_annotations(annfile_path, image_id):

    annotations = []
    with open(annfile_path, 'r') as file:
        filelines = file.readlines()
    instances = [line.strip().split(',') for line in filelines]
    instances = np.array( [[int(el) for el in line if el != ''] for line in instances] )

    for i, instance in enumerate(instances):
        bbox = list(instance[:4])
        obj_class = instance[5]

        if obj_class not in IGNORED_VISDRONE_CATEGORIES:
            obj_dict = {
                'is_crowd': 0,
                'bbox': bbox,
                'category_id': VISDRONE_CATEGORY_IDS_TO_UAV[obj_class],
                'bbox_mode': BoxMode.XYWH_ABS,
                'target_id': int( str(image_id) + str(i) )
            }
            annotations.append(obj_dict)

    return annotations


def generate_visdrone_dataset_dictionaries(dataset_dir):

    result = []
    imgdir = join(dataset_dir, 'images')
    anndir = join(dataset_dir, 'annotations')
    imgnames = get_images_from_dir(imgdir)

    for imgname in imgnames:
        im_path = join(imgdir, imgname)
        im_pil = Image.open(im_path)
        width, height = im_pil.size
        image_id = imgname.strip('.jpg').split('_')
        image_id.pop(2)
        image_id = int(''.join(image_id))

        annfile_path = join(anndir, imgname.strip('.jpg') + '.txt')
        annotations = read_annotations(annfile_path, image_id)

        imgdict = {
            'file_name': im_path,
            'height': height,
            'width': width,
            'image_id': image_id,
            'annotations': annotations
        }
        result.append(imgdict)
    
    return result
