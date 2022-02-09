import logging
import numpy as np
import torch
from torch import nn
import time

from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN

from networks.selective_rpn import SelectiveRPN

SYNCHRONIZE = True

class SelectiveMaskRCNN(GeneralizedRCNN):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.proposal_generator = SelectiveRPN(cfg, self.backbone.output_shape())
        print(type(self.proposal_generator))

   
    def scan(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training
        
        if SYNCHRONIZE: torch.cuda.synchronize()

        images = self.preprocess_image(batched_inputs)
        
        #BACKBONE
        start = time.perf_counter()
        features = self.backbone(images.tensor)
        if SYNCHRONIZE: torch.cuda.synchronize()
        end = time.perf_counter()
        print("backbone time: ", end - start)

        if detected_instances is None:
            if self.proposal_generator:
                
                #RPN
                start = time.perf_counter()
                proposals = self.proposal_generator.gen_partial_proposals(images, features, None)
                if SYNCHRONIZE: torch.cuda.synchronize()
                end = time.perf_counter()
                print('proposals: ', len(proposals[0]), 'time: ', end - start)
            
            else: 
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            #MASK + CLASS
            start = time.perf_counter()
            results, _ = self.roi_heads(images, features, proposals, None)
            if SYNCHRONIZE: torch.cuda.synchronize()
            end = time.perf_counter()
            print('roi_heads time: ', end - start)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

