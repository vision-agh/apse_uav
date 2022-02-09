import cv2
import os
import numpy as np
import torch
import math
import random

from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.engine import DefaultPredictor
import detectron2.data.transforms as T
from detectron2.structures import ImageList


def get_images_from_dir(path):

    extensions = ['jpg', 'png', 'bmp']
    contents = os.listdir(path)
    contents.sort()
    image_contents = [imgpath for imgpath in contents if imgpath.split('.')[-1] in extensions and 'Annotated' not in imgpath]

    return image_contents


def draw_bboxes(img, cars_indices, BBOX_WIDTH):

    for r, c in zip(cars_indices[0], cars_indices[1]):
        start_point = (c - BBOX_WIDTH, r - BBOX_WIDTH)
        end_point = (c + BBOX_WIDTH, r + BBOX_WIDTH)
        img = cv2.rectangle(img, start_point, end_point, (0, 0, 255), 1)


def calculate_image_slices(img, patch_size):

    height = img.shape[0]
    width = img.shape[1]
    h_grid = int(np.floor(height/patch_size))
    w_grid = int(np.floor(width/patch_size))
    patch_coords = [0]*h_grid*w_grid

    i = 0
    for row in range(h_grid):
        for col in range(w_grid):
            patch_coords[i] = (row*patch_size, col*patch_size) 
            i += 1

    return patch_coords


def get_instances(cars_img, neg_img, bbox_width, patch_coords, patch_size):

        h_start = patch_coords[0]
        w_start = patch_coords[1]
        cars_patch = cars_img[h_start : h_start+patch_size, w_start : w_start+patch_size]
        # cv2.imshow('cars', cars_patch)
        neg_patch = neg_img[h_start : h_start+patch_size, w_start : w_start+patch_size]
        cars_indices = np.where(cars_patch[:, :, 2] == 255)
        neg_indices = np.where(neg_patch[:, :, 0] == 255)
        pos_bboxes = [[c - bbox_width, r - bbox_width, c + bbox_width, r + bbox_width] for r, c in zip(cars_indices[0], cars_indices[1])]
        neg_bboxes = [[c - bbox_width, r - bbox_width, c + bbox_width, r + bbox_width] for r, c in zip(neg_indices[0], neg_indices[1])]
        pos_classes = [1]*len(pos_bboxes)
        neg_classes = [0]*len(neg_bboxes)
        
        #crop bboxes to patch
        for b in range(len(pos_bboxes)):
            for c in range(len(pos_bboxes[b])):
                if pos_bboxes[b][c] < 0:
                    pos_bboxes[b][c] = 0
                if pos_bboxes[b][c] >= patch_size:
                    pos_bboxes[b][c] = patch_size

        for b in range(len(neg_bboxes)):
            for c in range(len(neg_bboxes[b])):
                if neg_bboxes[b][c] < 0:
                    neg_bboxes[b][c] = 0
                if neg_bboxes[b][c] >= patch_size:
                    neg_bboxes[b][c] = patch_size

        boxes = Boxes(torch.tensor(pos_bboxes + neg_bboxes).to(torch.device('cuda:0')) )
        gt_classes = torch.tensor(pos_classes + neg_classes).to(torch.device('cuda:0') )

        return boxes, gt_classes




class CowcDataloaderOld():

    def __init__(self, config, DATASET_DIR):

        self.IMG_PATCH_SIZE = config.INPUT.MIN_SIZE_TEST
        self.BATCH_SIZE = 100
        self.BBOX_WIDTH = 18
        self.DATASET_DIR = DATASET_DIR
        self.config = config
        self.predictor = DefaultPredictor(config)
        self.backbone = self.predictor.model.backbone
        self.box_pooler = self.predictor.model.roi_heads.box_pooler
        self.box_head = self.predictor.model.roi_heads.box_head
        
        self.batches_per_img = []
        print("Loading dataset...")
        imgnames = get_images_from_dir(self.DATASET_DIR)
        for imgname in imgnames:
            print(imgname)
            self.batches_per_img.append(self.get_batches_from_img(imgname))
        print('Dataset loaded.')


    def preprocess_image(self, original_image):

        device = torch.device(self.config.MODEL.DEVICE)
        transform_gen = T.ResizeShortestEdge(
            [self.config.INPUT.MIN_SIZE_TEST, self.config.INPUT.MIN_SIZE_TEST], self.config.INPUT.MAX_SIZE_TEST
        )
        pixel_mean = torch.Tensor(self.config.MODEL.PIXEL_MEAN).to(device).view(-1, 1, 1)
        pixel_std = torch.Tensor(self.config.MODEL.PIXEL_STD).to(device).view(-1, 1, 1)
        height, width = original_image.shape[:2]
        image = transform_gen.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
        batched_inputs = [inputs]

        images = [x["image"].to(device) for x in batched_inputs]
        images = [(x - pixel_mean) / pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


    def get_instances(self, cars_img, neg_img, bbox_width, patch_coords, patch_size):

        h_start = patch_coords[0]
        w_start = patch_coords[1]
        cars_patch = cars_img[h_start : h_start+patch_size, w_start : w_start+patch_size]
        # cv2.imshow('cars', cars_patch)
        neg_patch = neg_img[h_start : h_start+patch_size, w_start : w_start+patch_size]
        cars_indices = np.where(cars_patch[:, :, 2] == 255)
        neg_indices = np.where(neg_patch[:, :, 0] == 255)
        pos_bboxes = [[c - bbox_width, r - bbox_width, c + bbox_width, r + bbox_width] for r, c in zip(cars_indices[0], cars_indices[1])]
        neg_bboxes = [[c - bbox_width, r - bbox_width, c + bbox_width, r + bbox_width] for r, c in zip(neg_indices[0], neg_indices[1])]
        pos_classes = [1]*len(pos_bboxes)
        neg_classes = [0]*len(neg_bboxes)
        
        #crop bboxes to patch
        for b in range(len(pos_bboxes)):
            for c in range(len(pos_bboxes[b])):
                if pos_bboxes[b][c] < 0:
                    pos_bboxes[b][c] = 0
                if pos_bboxes[b][c] >= patch_size:
                    pos_bboxes[b][c] = patch_size - 1

        for b in range(len(neg_bboxes)):
            for c in range(len(neg_bboxes[b])):
                if neg_bboxes[b][c] < 0:
                    neg_bboxes[b][c] = 0
                if neg_bboxes[b][c] >= patch_size:
                    neg_bboxes[b][c] = patch_size - 1

        boxes = Boxes(torch.tensor(pos_bboxes + neg_bboxes).to(torch.device('cuda:0')) )
        gt_classes = torch.tensor(pos_classes + neg_classes).to(torch.device('cuda:0') )

        return boxes, gt_classes


    def get_batches_from_img(self, imgname):

        cars_imgname = imgname.split('.')[0] + '_Annotated_Cars.png'
        neg_imgname = imgname.split('.')[0] + '_Annotated_Negatives.png'
        cars_img = cv2.imread(os.path.join(self.DATASET_DIR, cars_imgname))
        neg_img = cv2.imread(os.path.join(self.DATASET_DIR, neg_imgname))
        bigimg = cv2.imread(os.path.join(self.DATASET_DIR, imgname))
        patches_startpoints = calculate_image_slices(bigimg, self.IMG_PATCH_SIZE)
        train_features = None
        train_classes = None

        for patch_coords in patches_startpoints:
            h_start = patch_coords[0]
            w_start = patch_coords[1]
            img_patch = bigimg[h_start : h_start+self.IMG_PATCH_SIZE, w_start : w_start+self.IMG_PATCH_SIZE]
            boxes, gt_classes = get_instances(cars_img, neg_img, self.BBOX_WIDTH, patch_coords, self.IMG_PATCH_SIZE)
            if len(gt_classes) > 0:
                with torch.no_grad():
                    batched_input = self.preprocess_image(img_patch)
                    features = self.backbone(batched_input.tensor)
                    features = [features[f] for f in self.config.MODEL.ROI_HEADS.IN_FEATURES]
                    roi_features = self.box_pooler(features, [boxes])
                    conv_roi_features = self.box_head(roi_features)
                if train_features is None:
                    train_features = conv_roi_features.cpu()
                    train_classes = gt_classes.cpu()
                else:
                    train_features = torch.cat([train_features, conv_roi_features.cpu()], 0)
                    train_classes = torch.cat([train_classes, gt_classes.cpu()], 0)

        pos_indices = torch.where(train_classes == 1)[0]
        neg_indices = torch.where(train_classes == 0)[0]
        num_pos = len(pos_indices)
        num_neg = len(neg_indices)
        paired_num = min(num_pos, num_neg)
        num_samples = int(self.BATCH_SIZE/2)
        num_batches = math.floor(paired_num / num_samples)

        batches_list = []
        for batch_idx in range(num_batches):

            batch_pos_indices = pos_indices[batch_idx*num_samples : (batch_idx+1)*num_samples]
            batch_neg_indices = neg_indices[batch_idx*num_samples : (batch_idx+1)*num_samples]
            pos_features = train_features[batch_pos_indices]
            neg_features = train_features[batch_neg_indices]
            batch_train_features = torch.cat([pos_features, neg_features], 0)
            batch_train_labels = torch.tensor([1]*num_samples + [0]*num_samples)
            batches_list.append( (batch_train_features, batch_train_labels) )

        return batches_list


class CowcDataloader():

    def __init__(self, config, DATASET_DIR):

        self.IMG_PATCH_SIZE = config.INPUT.MIN_SIZE_TEST
        self.BATCH_SIZE = 100
        self.BBOX_WIDTH = 18
        self.IMAGES_PER_BATCH = 2
        self.DATASET_DIR = DATASET_DIR
        self.config = config
        self.imgnames = get_images_from_dir(self.DATASET_DIR)
        self.current_img = None
        self.current_cars_img = None
        self.current_patches_startpoints = None
        self.current_instances_per_patch = None
        

    def preprocess_image(self, original_image):

        device = torch.device(self.config.MODEL.DEVICE)
        transform_gen = T.ResizeShortestEdge(
            [self.config.INPUT.MIN_SIZE_TEST, self.config.INPUT.MIN_SIZE_TEST], self.config.INPUT.MAX_SIZE_TEST
        )
        pixel_mean = torch.Tensor(self.config.MODEL.PIXEL_MEAN).to(device).view(-1, 1, 1)
        pixel_std = torch.Tensor(self.config.MODEL.PIXEL_STD).to(device).view(-1, 1, 1)
        height, width = original_image.shape[:2]
        image = transform_gen.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
        batched_inputs = [inputs]

        images = [x["image"].to(device) for x in batched_inputs]
        images = [(x - pixel_mean) / pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


    def load_img(self, imgname):

        cars_imgname = imgname.split('.')[0] + '_Annotated_Cars.png'
        self.current_img = cv2.imread(os.path.join(self.DATASET_DIR, imgname) )
        self.current_cars_img = cv2.imread(os.path.join(self.DATASET_DIR, cars_imgname) )
        self.current_patches_startpoints = calculate_image_slices(self.current_img, self.IMG_PATCH_SIZE)
        self.current_instances_per_patch = {}
        for patch_coords in self.current_patches_startpoints:
            h_start = patch_coords[0]
            w_start = patch_coords[1]
            img_patch = self.current_img[h_start : h_start+self.IMG_PATCH_SIZE, w_start : w_start+self.IMG_PATCH_SIZE]
            cars_patch = self.current_cars_img[h_start : h_start+self.IMG_PATCH_SIZE, w_start : w_start+self.IMG_PATCH_SIZE]
            cars_indices = np.where(cars_patch[:, :, 2] == 255) # tuple of two arrays
            if len(cars_indices[0]) > 0:   
                patch_instances = Instances( (self.IMG_PATCH_SIZE, self.IMG_PATCH_SIZE) )
                pos_bboxes = [[c - self.BBOX_WIDTH, r - self.BBOX_WIDTH, c + self.BBOX_WIDTH, r + self.BBOX_WIDTH] for r, c in zip(cars_indices[0], cars_indices[1])]
                classes = [0]*len(pos_bboxes)
                for b in range(len(pos_bboxes)):
                    for c in range(len(pos_bboxes[b])):
                        if pos_bboxes[b][c] < 0:
                            pos_bboxes[b][c] = 0
                        if pos_bboxes[b][c] >= self.IMG_PATCH_SIZE:
                            pos_bboxes[b][c] = self.IMG_PATCH_SIZE - 1
                gt_boxes = Boxes(torch.tensor(pos_bboxes).to(torch.device('cuda:0')) )
                gt_classes = torch.tensor(classes).to(torch.device('cuda:0') )
                patch_instances._fields['gt_boxes'] = gt_boxes
                patch_instances._fields['gt_classes'] = gt_classes
                self.current_instances_per_patch[patch_coords] = patch_instances


    def get_batch(self):

        batch = []
        for i in range(self.IMAGES_PER_BATCH):
            print(patches_startpoints)


    # def get_imgdict(self, patch_coords):



           