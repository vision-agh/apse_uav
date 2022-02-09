import cv2
import os
import numpy as np
import torch
import math
import random
import re
from pycocotools.coco import COCO
import json

from detectron2.structures.boxes import Boxes, BoxMode
from detectron2.structures.instances import Instances
import detectron2.data.transforms as T


COCO_CATEGORY_IDS_TO_UAV = {
    1: 3,   # coco person
    3: 0,   # coco car
    6: 2,   # coco bus
    8: 1    # coco truck
}


def generate_coco_dataset_dictionaries(json_filepath, imgfolder_path, allowed_classes=None, category_mapping=None, precomputed_proposals=False):
    # read coco dataset and convert it to detectron2 dataset dicts format

    result_dicts = []
    anns_per_image_id = {}

    print('Reading coco dataset json...')
    with open(json_filepath, 'r') as json_file:
        coco_dict = json.load(json_file)

    if allowed_classes:
        print('Picking images with', allowed_classes, 'objects...')
        allowed_category_ids = [category['id'] for category in coco_dict['categories'] if category['name'] in allowed_classes]
        allowed_annotations = [ann for ann in coco_dict['annotations'] if ann['category_id'] in allowed_category_ids and ann['iscrowd'] == 0]    
    else:
        allowed_annotations = [ann for ann in coco_dict['annotations'] if ann['iscrowd'] == 0]

    # prepare annotations in detectron2 format
    for ann in allowed_annotations:
        annotation_dict = {
            'is_crowd': ann['iscrowd'],
            'bbox': ann['bbox'],
            'category_id': category_mapping[ann['category_id']] if category_mapping else ann['category_id'],
            'bbox_mode': BoxMode.XYWH_ABS,
            'target_id': ann['id']
        }
        if 'segmentation' in ann:
            annotation_dict['segmentation'] = ann['segmentation'],
        if ann['image_id'] in anns_per_image_id:
            anns_per_image_id[ann['image_id']].append(annotation_dict)
        else:
            anns_per_image_id[ann['image_id']] = [annotation_dict]

    allowed_image_ids = [ann['image_id'] for ann in allowed_annotations]
    allowed_image_ids = np.unique(allowed_image_ids)
    allowed_images = [img_dict for img_dict in coco_dict['images'] if img_dict['id'] in allowed_image_ids]
    
    # prepare img dicts in detectron2 format
    for img_dict in allowed_images:
        result_img_dict = {
            'file_name': os.path.join(imgfolder_path, img_dict['file_name']),
            'height': img_dict['height'],
            'width': img_dict['width'],
            'image_id': img_dict['id'],
            'annotations': anns_per_image_id[img_dict['id']]
        }
        if precomputed_proposals:
            result_img_dict['proposal_boxes'] = np.array( [ann['bbox'] for ann in result_img_dict['annotations']] )
            result_img_dict['proposal_bbox_mode'] = BoxMode.XYWH_ABS
            result_img_dict['proposal_objectness_logits'] = np.array( [1] * len(result_img_dict['proposal_boxes']) )
        result_dicts.append(result_img_dict)

    return result_dicts
        

def detectron2_dataset_to_coco(img_dicts, categories):

    coco_gt = COCO()
    dataset = {
        'images': [],
        'annotations': [],
        'categories': []
    }

    for i, cat in enumerate(categories):
        category_dict = {
            'id': i,
            'name': cat
        }
        dataset['categories'].append(category_dict)

    for img_dict in img_dicts:
        image_info = {
            'id': img_dict['image_id'],
            'file_name': os.path.basename(img_dict['file_name']),
            'width': img_dict['width'],
            'height': img_dict['height']
        }
        dataset['images'].append(image_info)

        for instance in img_dict['annotations']:
            coco_dict = {
                'iscrowd': instance['is_crowd'],
                'image_id': img_dict['image_id'],
                'category_id': instance['category_id'],
                'bbox': instance['bbox'],
                'id': instance['target_id'],
                'area': instance['bbox'][2] * instance['bbox'][3] # XYWH bbox format
            }
            if 'segmentation' in instance:
                coco_dict['segmentation'] = instance['segmentation']
            dataset['annotations'].append(coco_dict)

    coco_gt.dataset = dataset
    coco_gt.createIndex()
    return coco_gt



class COCODataloader():

    def __init__(self, config, DATASET_DIR, imgs_per_batch = 2, categories = ['car'], version = 'train2017'):

        self.config = config
        self.DATASET_DIR = DATASET_DIR
        self.imgs_per_batch = imgs_per_batch
        print(type(self.imgs_per_batch))
        self.dataset_version = version
        self.annotation_file = os.path.join(DATASET_DIR, 'annotations', 'instances_' + version + '.json')
        self.coco_dataset = COCO(self.annotation_file)
        self.categories_ids = self.coco_dataset.getCatIds(catNms = categories)
        self.images_ids = self.coco_dataset.getImgIds(catIds = self.categories_ids)
        self.images_dicts = self.coco_dataset.loadImgs(self.images_ids)


    def __iter__(self):

        return self


    def __next__(self):

        result = []
        for i in range(self.imgs_per_batch):
            random_index = random.randrange(0, len(self.images_dicts))
            result += self.get_instances_from_imagedict(self.images_dicts[random_index])
        return result

    
    def get_instances_from_imagedict(self, imgdict):

        img_anns_id = self.coco_dataset.getAnnIds(imgIds=imgdict['id'], catIds=self.categories_ids, iscrowd=None)
        img_anns = self.coco_dataset.loadAnns(img_anns_id)
        boxes = [ann['bbox'] for ann in img_anns]
        boxes = np.array([ [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]] for bbox in boxes ] ).astype(float)
        masks = [self.coco_dataset.annToMask(ann) for ann in img_anns]
        imgpath = os.path.join(self.DATASET_DIR, self.dataset_version, imgdict['file_name'])
        img = cv2.imread(imgpath)
        frame_instances = Instances( (img.shape[0], img.shape[1]) )
        # cv2.imshow('test', img)
        # for i, mask in enumerate(masks):
        #     cv2.imshow(str(boxes[i]), mask*255)
        gt_classes = torch.Tensor([0] * len(boxes))
        frame_instances._fields['gt_classes'] = gt_classes
        frame_instances._fields['gt_boxes'] = boxes
        frame_instances._fields['gt_masks'] = masks
        image_batch = self.preprocess_image_and_instances(img, frame_instances)
        image_batch[0]['file_name'] = imgpath
        
        return image_batch


    def preprocess_image_and_instances(self, original_image, instances):

        with torch.no_grad():
            device = torch.device(self.config.MODEL.DEVICE)
            transform_gen = T.ResizeShortestEdge(
                [self.config.INPUT.MIN_SIZE_TEST, self.config.INPUT.MIN_SIZE_TEST], self.config.INPUT.MAX_SIZE_TEST
            )
            pixel_mean = torch.Tensor(self.config.MODEL.PIXEL_MEAN).to(device).view(-1, 1, 1)
            pixel_std = torch.Tensor(self.config.MODEL.PIXEL_STD).to(device).view(-1, 1, 1)
            height, width = original_image.shape[:2]
            image = transform_gen.get_transform(original_image).apply_image(original_image)
            transform = transform_gen.get_transform(original_image)

            result_boxes = transform.apply_box(instances.gt_boxes)
            result_boxes = Boxes( torch.Tensor(result_boxes) )
            instances.gt_boxes = result_boxes

            # print('przed:')
            # print(instances.gt_masks)
            result_masks = instances.gt_masks
            for i, mask in enumerate(result_masks):
                result_masks[i] = transform.apply_segmentation(mask)
            result_masks = torch.Tensor(result_masks)
            instances.gt_masks = result_masks
            # print('po:')
            # print(instances.gt_masks)

            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width, "instances": instances}
            batched_inputs = [inputs]

            return batched_inputs