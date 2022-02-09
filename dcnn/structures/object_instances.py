import itertools
import torch
import numpy as np
import sys
import time
from typing import Any, Dict, List, Tuple, Union
from detectron2.structures.instances import Instances

from structures.set_boxes import SetBoxes

class ObjectInstances(Instances):

    def __init__(self, image_size: Tuple[int, int], display_info: List=[], metadata: Any=None, **kwargs: Any):
        super().__init__(image_size=image_size, **kwargs)
        self._assigned_ids = []
        self._display_info = display_info
        self._metadata = metadata
        self._num_objects = 0


    def __len__(self) -> int:
        # check if empty
        if not self._fields:
            return 0
        else:
            for v in self._fields.values():
                return len(v)
        
        return self._num_objects


    def __str__(self):
        s = 'objects: ' + str(len(self))
        s += '\n'
        for object_idx in range(0, len(self)):
            if self._metadata is not None:
                s += '\tid: {}'.format(self.ids[object_idx]) + '\tclass: ' + self._metadata.get("thing_classes", None)[self.pred_classes[object_idx].item()]  + '\tundetected for: {}\n'.format(self.frames_since_detected[object_idx])
            else:
                s += '\tid: {}\tclass: {}\tundetected for: {}\n'.format(
                    self.ids[object_idx],
                    self.pred_classes[object_idx].item(),
                    self.frames_since_detected[object_idx]
                    )

        return s


    def get_new_id(self):

        if len(self._assigned_ids) == 0:
            return 1
        return self._assigned_ids[-1] + 1

    
    # tenosr.to method
    def to(self, device: str) -> "Instances":
        """
        Returns:
            Instances: all fields are called with a `to(device)`, if the field has this method.
        """
        ret = ObjectInstances(self._image_size)
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            ret.set(k, v)
        return ret


    def add_new_object(self, detection_id, detections, detection_embeddings=None, verbose=True):
        # ['pred_boxes', 'scores', 'pred_classes', 'pred_masks'])
        # print("new object, detection_id: ", detection_id)detections_dict['pred_boxes'][detection_id]
        detections_dict = detections.get_fields()
        objects_dict = self._fields
        new_id = self.get_new_id()
        if verbose:
            if 'new_objects' in self._display_info: print('adding detection_id: {} as new object with id: {}'.format(detection_id, new_id))
        if len(self) == 0:
            objects_dict['detected_this_frame'] = [True]
            objects_dict['ids'] = [new_id]
            objects_dict['frames_since_detected'] = [0]
            objects_dict['pred_boxes'] = [ SetBoxes(detections_dict['pred_boxes'][detection_id].tensor) ]
            objects_dict['scores'] = [ detections_dict['scores'][detection_id] ]
            objects_dict['pred_classes'] = [ detections_dict['pred_classes'][detection_id] ]
            new_mask = detections_dict['pred_masks'][detection_id]
            objects_dict['pred_masks'] = new_mask.view(1, new_mask.size()[0], new_mask.size()[1])
            # print('pred masks size:', objects_dict['pred_masks'].size())
            if detection_embeddings is not None:
                objects_dict['embeddings'] = [ detection_embeddings[detection_id] ]
        else:
            objects_dict['detected_this_frame'].append(True)
            objects_dict['ids'].append(new_id)
            objects_dict['frames_since_detected'].append(0)
            objects_dict['pred_boxes'].append( SetBoxes(detections_dict['pred_boxes'][detection_id].tensor) )
            objects_dict['scores'].append(detections_dict['scores'][detection_id])
            objects_dict['pred_classes'].append(detections_dict['pred_classes'][detection_id])
            new_mask = detections_dict['pred_masks'][detection_id]
            objects_dict['pred_masks'] = torch.cat( (objects_dict['pred_masks'], new_mask.view(1, new_mask.size()[0], new_mask.size()[1]) ) )
            # print('pred masks size:', objects_dict['pred_masks'].size())
            if detection_embeddings is not None:
                objects_dict['embeddings'].append(detection_embeddings[detection_id])

        self._assigned_ids.append(new_id)

    
    def delete_undetected_objects(self, frames_threshold):

        if len(self) > 0:
            indexes_to_delete = []
            for object_index in range(len(self)):
                if self.frames_since_detected[object_index] > frames_threshold:
                    indexes_to_delete.append(object_index)

            indexes_to_delete = sorted(indexes_to_delete, reverse=True)
            torch_indexes_to_keep = torch.tensor([ i not in indexes_to_delete for i in range(self.pred_masks.size()[0]) ])
            self._fields['pred_masks'] = self._fields['pred_masks'][torch_indexes_to_keep]

            for index in indexes_to_delete:
                del self._fields['detected_this_frame'][index]
                del self._fields['ids'][index]
                del self._fields['frames_since_detected'][index]
                del self._fields['pred_boxes'][index]
                del self._fields['scores'][index]
                del self._fields['pred_classes'][index]
                if 'embeddings' in self._fields:
                    del self.embeddings[index]


    #debug
    def check_lengths(self):

        first_value = list( self._fields.values() )[0]
        prev_len = len(first_value)
        for k, v in self._fields.items():
            if len(v) != prev_len:
                print('key {} has length {} (prev {})'.format(k, len(v), prev_len))
                sys.exit()
            prev_len = len(v)
    

    def associate_detection(self, detection_id, object_index, detections, detections_embeddings=None):

        if 'associations' in self._display_info:
            print('associating detection {} to object id: {}'.format(detection_id, self.ids[object_index]))
        detections_dict = detections.get_fields()
        objects_dict = self.get_fields()
        objects_dict['detected_this_frame'][object_index] = True
        objects_dict['frames_since_detected'][object_index] = 0
        objects_dict['pred_boxes'][object_index] = detections_dict['pred_boxes'][detection_id]
        objects_dict['pred_classes'][object_index] = detections_dict['pred_classes'][detection_id]
        objects_dict['pred_masks'][object_index] = detections_dict['pred_masks'][detection_id]
        if 'embeddings' in self._fields and detections_embeddings is not None:
            self.embeddings[object_index] = detections_embeddings[detection_id]

    
    def finish_association(self):
        """should be called at the end of the frame, updates dictionary['frames_since_detected']"""
        self.frames_since_detected = [self.frames_since_detected[obj_index] + 1 
                                        if not self.detected_this_frame[obj_index] 
                                        else 0 
                                        for obj_index in range(len(self)) ]

        self.detected_this_frame = [False] * len(self)
            

    def get_recent_objects(self):

        new_objects = ObjectInstances(image_size=self._image_size, display_info=self._display_info, metadata=self._metadata)
        for obj_idx in range(len(self)):
            if self.detected_this_frame[obj_idx]:
                if len(new_objects) == 0:
                    for k in self._fields.keys():
                        new_objects._fields[k] = [ self._fields[k][obj_idx] ]
                else:
                    for k in self._fields.keys():
                        new_objects._fields[k].append(self._fields[k][obj_idx])

        return new_objects