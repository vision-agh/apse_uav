import torch
import numpy as np
from torchvision.ops import roi_pool, roi_align
import torch.nn.functional as F
from pycocotools.mask import decode

from detectron2.modeling.backbone import build_backbone
from detectron2.structures import ImageList
import detectron2.data.transforms as T
from detectron2.structures.boxes import Boxes

from utils.partial_checkpointer import PartialCheckpointer
from utils.mask_utils import show_mask


class RoiFeaturesGenerator():

    """
    Generates roi aligned features from ground truth video frame.
    To be used in association head training
    Uses tracker's backbone and roi align head
    """

    def __init__(self, config, roi_size=8):

        self.device = torch.device(config.MODEL.DEVICE)

        self.roi_size = roi_size
        self.in_features = config.MODEL.ROI_HEADS.IN_FEATURES
        self.backbone = build_backbone(config)
        self.backbone.to(self.device)
        # self.roi_pooler = self.init_roi_pooler(config, roi_size)

        checkpointer = PartialCheckpointer(self.backbone)
        checkpointer.load(config.MODEL.WEIGHTS)

        self.transform_gen = T.ResizeShortestEdge(
            [config.INPUT.MIN_SIZE_TEST, config.INPUT.MIN_SIZE_TEST], config.INPUT.MAX_SIZE_TEST
        )

        self.pixel_mean = torch.Tensor(config.MODEL.PIXEL_MEAN).to(self.device).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(config.MODEL.PIXEL_STD).to(self.device).view(-1, 1, 1)

    
    # def init_roi_pooler(self, config, roi_size):

    #     input_shape = self.backbone.output_shape()
    #     print(input_shape)
    #     # pooler_resolution        = config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
    #     # pooler_scales            = tuple(1.0 / input_shape[k].stride for k in self.in_features)
    #     # sampling_ratio           = config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
    #     # pooler_type              = config.MODEL.ROI_BOX_HEAD.POOLER_TYPE

    #     # return roi_pool(output_size)


    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        inputs = {"image": image, "height": height, "width": width}
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


    def get_rois_features(self, original_image, objects, objects_masks=None):
        """
        args:
            original_image - frame from sequence
            objects - object instances from frame
        return:
            np.array of ids, torch.Tensor of frame rois of shape (N, C, roi_size, roi_size)
        """

        ## objects row: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>
        height, width = original_image.shape[:2]
        image = self.transform_gen.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
        image = self.preprocess_image([inputs])
        bbox_coords = objects[:, 2:6]
        # print('image tensor: ', image.tensor.size())
        # print('image tensor: ', image.tensor[0, 0, :, :])
        # print('backbone weights:')
        # for i, param in enumerate( self.backbone.parameters() ):
        #     if i < 3:
        #         print(param.data)

        ## boxes (x1, y1, x2, y2) are in original_image coordinates
        boxes = torch.Tensor( [ [0, bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]] for i, bbox in enumerate(bbox_coords) ] )
        features = self.backbone(image.tensor)[self.in_features[0]].cpu()
        if objects_masks is not None:
            masks = []
            for mask_rle in objects_masks:
                bin_mask = torch.tensor(decode(mask_rle) )
                masks.append(bin_mask)
            masks = torch.stack(masks)
            resized_masks = F.interpolate(masks.float().view(-1, 1, masks.size()[1], masks.size()[2]),
                                            size=(features.size()[2], features.size()[3]),
                                            mode='bilinear')
            expanded_masks = resized_masks.expand(resized_masks.size()[0], features.size()[1], features.size()[2], features.size()[3])
            expanded_features = features.expand(resized_masks.size()[0], features.size()[1], features.size()[2], features.size()[3]).clone()
            cropped_features = torch.mul(expanded_features, expanded_masks)
            batch_indices = torch.tensor(range(cropped_features.size()[0]))
            boxes[:, 0] = batch_indices
        ## spatial scale: features_width/boxes_scale(original_image_width)
        spatial_scale = features.size()[3] / width
        if objects_masks is not None:
            rois = roi_align(cropped_features, boxes, (self.roi_size, self.roi_size), spatial_scale, sampling_ratio=4)
        else:
            rois = roi_pool(features, boxes, (self.roi_size, self.roi_size), spatial_scale)
        # show_mask(rois[0, 0], 'roi')
        # show_mask(masks[0], 'mask')
  
        return torch.Tensor(objects[:, 1]).to(self.device), rois.to(self.device)


    # def boxes_from_objects(self, objects):

    #     bbox_coords = objects[:, 2:6]
    #     # change to (x1, y1, x2, y2) format
    #     boxes = torch.Tensor( [ [0, bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]] for i, bbox in enumerate(bbox_coords) ] ).to(self.device)

    #     return boxes


    def get_features_depth(self):

        return self.backbone.output_shape()[self.in_features[0]].channels