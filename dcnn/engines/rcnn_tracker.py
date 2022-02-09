import cv2
import argparse
import torch
torch.set_printoptions(profile="full")
import time
import numpy as np
import sys
import string
from torchvision.ops import roi_pool, roi_align
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.engine.defaults import DefaultPredictor

from engines.track_predictor import TrackPredictor
from structures.object_instances import ObjectInstances
from structures.set_boxes import SetBoxes
from utils.track_visualizer import TrackVisualizer
from utils.mask_utils import(
    get_mask_centroid,
    show_mask,
    compute_masks_iou
) 
from networks.association_head import AssociationHead


## TODO - put them in config
ASSOCIATION_ROI_SIZE = 10
ASSOCIATION_WEIGHTS_PATH = '/home/magister/magisterka/rcnn_tracker/pretrained/association_head_UAVCHECKPOINT_R_101_FPN_3x/association_head_EP2.pth'
#association_metric

class RcnnTracker:

    def __init__(self, config, image_size, weights, association_metric='embeddings', DISPLAY_INFO=[], metadata=None):

        #['objects, 'frame_count', 'detections', 'new_objects', 'associations', 'hungarian_matches', 'recent_objects']
        self.metadata = metadata
        self.DISPLAY_INFO = DISPLAY_INFO
        self.association_metric = association_metric
        self.MASKS_IOU_THRESHOLD = 0.7
        self.ASSOCIATION_EMBEDDING_THRESHOLD = 0.6        ## TODO - put it in config
        self.OBJECT_UNDETECTED_FRAMES_TH = 100            ## TODO - put it in config
        self.crop_features = False
        self.config = config
        self.image_size = image_size
        self.device = torch.device(config.MODEL.DEVICE)
        self.predictor = TrackPredictor(self.config)
        self.backbone_features_depth = self.predictor.model.backbone.output_shape()[config.MODEL.ROI_HEADS.IN_FEATURES[0]].channels
        
        self.association_head = AssociationHead(roi_size=ASSOCIATION_ROI_SIZE, input_depth=self.backbone_features_depth)
        self.association_head.load_state_dict(torch.load(weights))
        self.association_head.to(self.device)

        self.objects = ObjectInstances(image_size=image_size, display_info=self.DISPLAY_INFO, metadata=metadata)
        self.frame_count = 0


    def next_frame(self, frame):

        self.frame_count += 1
        if 'frame_count' in self.DISPLAY_INFO: print("\nFRAME: ", self.frame_count)
        detections, backbone_features = self.predictor(frame)
        detections = detections['instances']
        self.associate_detections_to_objects(detections, backbone_features=backbone_features, metric='embeddings')
        self.objects.delete_undetected_objects(self.OBJECT_UNDETECTED_FRAMES_TH)
        if 'objects' in self.DISPLAY_INFO: print(self.objects)
        recent_objects = self.objects.get_recent_objects()
        if 'recent_objects' in self.DISPLAY_INFO: print('RECENT OBJECTS:\n', recent_objects)
        self.objects.finish_association()
        
        return recent_objects


    def associate_detections_to_objects(self, detections, backbone_features=None, metric='embeddings'):

        objects_dict = self.objects.get_fields()
        if 'detections' in self.DISPLAY_INFO: 
            print(len(detections), ' detections:')
            for detection_id in range(len(detections)):
                if self.metadata is not None:
                    print('detection_id: ', detection_id, 'class: ', self.metadata.get("thing_classes", None)[detections.pred_classes[detection_id].item()] )
                else:
                    print('detection_id: ', detection_id, 'class: ', detections.pred_classes[detection_id].item())

        # bbox = [x1, y1, x2, y2]
        if metric == 'bbox_center_dist':
            for detection_id in range(len(detections)):
                detection_bbox = detections.pred_boxes[detection_id]
                detection_bbox_center = detection_bbox.get_centers()
                detection_associated = False
                for object_index in range(len(self.objects)):
                    object_bbox = self.objects.pred_boxes[object_index]
                    object_bbox_center = object_bbox.get_centers()
                    # print('object bbox: ', type(object_bbox), object_bbox_center)
                    dist = torch.sum((detection_bbox_center - object_bbox_center)**2)
                    if dist < BBOX_CENTER_DIST_THRESHOLD:
                        detection_associated = True
                        self.objects.associate_detection(detection_id, object_index, detections)
                
                if not detection_associated:
                    self.objects.add_new_object(detection_id, detections)
        
        elif metric == 'mask_iou':
            for detection_id in range(len(detections)):
                if len(self.objects) == 0:
                    self.objects.add_new_object(detection_id, detections)
                else:
                    detection_mask = detections.pred_masks[detection_id]
                    detection_centroid = get_mask_centroid(detection_mask)
                    ious_with_objects = [compute_masks_iou(detection_mask, object_mask, detection_centroid) for object_mask in self.objects.pred_masks]
                    match_index = np.argmax(ious_with_objects)
                    if ious_with_objects[match_index] >= self.MASKS_IOU_THRESHOLD:
                        self.objects.associate_detection(detection_id, match_index, detections)
                    else:
                        self.objects.add_new_object(detection_id, detections)
        
        elif metric == 'embeddings':
            if len(detections) > 0:
                detection_rois = self.get_features_rois(detections, backbone_features, crop_features=self.crop_features)
                detection_embeddings = self.association_head(detection_rois)
                if len(self.objects) == 0:
                    for detection_id in range(len(detections)):
                        self.objects.add_new_object(detection_id, detections, detection_embeddings)
                else:
                    distances = self.calculate_distance_matrix(detection_embeddings)
                    ##linear_sum_assignment known as Hungarian algorithm
                    match_obj_indexes, match_det_indexes = linear_sum_assignment(distances.cpu().detach().numpy())
                    matched_detections = []
                    if 'hungarian_matches' in self.DISPLAY_INFO: print('\nhungarian matches:')
                    ## match current objects to new detections
                    for obj_idx, det_idx in zip(match_obj_indexes, match_det_indexes):
                        if 'hungarian_matches' in self.DISPLAY_INFO: print('obj {} to det {}'.format(obj_idx, det_idx))
                        obj_idx = int(obj_idx)
                        det_idx = int(det_idx)
                        dist = distances[obj_idx, det_idx]
                        if dist < self.ASSOCIATION_EMBEDDING_THRESHOLD:
                            self.objects.associate_detection(det_idx, obj_idx, detections, detection_embeddings)
                            matched_detections.append(det_idx)
                    ## add unmatched detections as new objects
                    for detection_id in range(len(detections)):
                        if detection_id not in matched_detections:
                            self.objects.add_new_object(detection_id, detections, detection_embeddings)
        

    def reset_tracker(self):

        self.objects = ObjectInstances(image_size=image_size, display_info=self.DISPLAY_INFO)
        self.frame_count = 0

    
    def get_features_rois(self, detections, backbone_features, crop_features=True):

        ## FEATURES SIZE: [1, <CHANNELS>, H, W]
        ## MASKS SIZE: [<NUM_DETECTIONS>, imH, imW]
        ## RESIZED MASKS SIZE: [<NUM_DETECTIONS>, H, W]
        ## EXPANDED MASKS SIZE: [<NUM_DETECTIONS>, <CHANNELS>, H, W] /copy along channels
        ## EXPANDED FEATURES SIZE: [<NUM_DETECTIONS>, <CHANNELS>, H, W] /copy along detections

        features = backbone_features[self.config.MODEL.ROI_HEADS.IN_FEATURES[0]]
        spatial_scale = features.size()[3] / self.image_size[1]
        if crop_features:
            resized_masks = F.interpolate(detections.pred_masks.float().view(-1, 1, detections.pred_masks.size()[1], detections.pred_masks.size()[2]),
                                            size=(features.size()[2], features.size()[3]),
                                            mode='bilinear')                      
            expanded_masks = resized_masks.expand(resized_masks.size()[0], features.size()[1], features.size()[2], features.size()[3])
            expanded_features = features.expand(resized_masks.size()[0], features.size()[1], features.size()[2], features.size()[3]).clone()
            cropped_features = torch.mul(expanded_features, expanded_masks)
            batch_indices = torch.tensor(range(cropped_features.size()[0])).view(-1, 1).to(self.device)
        else:
            batch_indices = torch.zeros(detections.pred_boxes.tensor.size()[0], 1).to(self.device)
        
        boxes = torch.cat([batch_indices, detections.pred_boxes.tensor], dim=1)

        if crop_features:
            rois = roi_align(cropped_features, boxes, (ASSOCIATION_ROI_SIZE, ASSOCIATION_ROI_SIZE), spatial_scale, sampling_ratio=4)
        else:
            rois = roi_pool(features, boxes, (ASSOCIATION_ROI_SIZE, ASSOCIATION_ROI_SIZE), spatial_scale)    

        ## DEBUG
        # print('first obj, first channel:', rois[0, 0])
        # show_mask(rois[0, 0])
        # show_mask(detections.pred_masks[0], text='obj mask')

        return rois


    def calculate_distance_matrix(self, detection_embeddings):
       
        OBJECTS_embeddings = torch.cat( [ torch.stack( [self.objects.embeddings[i] ] * len(detection_embeddings) )
                                                for i in range(len(self.objects)) ] ) 
        DETECTIONS_embeddings =  torch.cat( [ detection_embeddings ] * len(self.objects) )
        DIFFS_embeddings = OBJECTS_embeddings - DETECTIONS_embeddings                                                                       
        embedding_dim = DIFFS_embeddings.size()[1]
        num_distances = DIFFS_embeddings.size()[0]
        ## row-wise vector product of DIFFS_embeddings by DIFFS_embeddings
        distances = torch.bmm( DIFFS_embeddings.view(num_distances, 1, embedding_dim), DIFFS_embeddings.view(num_distances, embedding_dim, 1) )
        
        ##debug
        # ob1_det1_diff = self.objects.embeddings[0] - detection_embeddings[0]
        # ob1_to_det1 = torch.dot(ob1_det1_diff, ob1_det1_diff)

        # ob1_det2_diff = self.objects.embeddings[0] - detection_embeddings[1]
        # ob1_to_det2 = torch.dot(ob1_det2_diff, ob1_det2_diff)

        # print('ob1todet1: ', ob1_to_det1)
        # print('ob1todet2: ', ob1_to_det2)

        # dist = torch.zeros(len(self.objects), len(detection_embeddings) )
        # for o in range(len(self.objects)):
        #     for d in range(len(detection_embeddings)):
        #         dist[o, d] = torch.dot( self.objects.embeddings[o] - detection_embeddings[d], self.objects.embeddings[o] - detection_embeddings[d] )
        # print('dist: ', dist)

        distance_matrix = distances.view(len(self.objects), len(detection_embeddings))
        
        return distance_matrix
