import torch
import time

from detectron2.modeling.proposal_generator import RPN
from detectron2.modeling.proposal_generator.rpn_outputs import RPNOutputs, find_top_rpn_proposals

class SelectiveRPN(RPN):

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)


    
    def gen_partial_proposals(self, images, features, gt_instances=None):
        """
        Args:
            images (ImageList): input images of length `N`
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
                Each `Instances` stores ground-truth instances for the corresponding image.

        Returns:
            proposals: list[Instances]: contains fields "proposal_boxes", "objectness_logits"
            loss: dict[Tensor] or None
        """
        features = [features[f] for f in self.in_features]
        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        anchors = self.anchor_generator(features)

        outputs = RPNOutputs(
            self.box2box_transform,
            self.batch_size_per_image,
            images,
            pred_objectness_logits,
            pred_anchor_deltas,
            anchors,
            None,
            None,
            self.smooth_l1_beta,
        )

        # predicted_proposals = outputs.predict_proposals()
        # predicted_objectness = outputs.predict_objectness_logits()
        predicted_proposals = [ outputs.predict_proposals()[-1] ]
        predicted_objectness = [ outputs.predict_objectness_logits()[-1] ]
        #217413

        torch.cuda.synchronize()


        print('\npredicted_proposals type: ', type(predicted_proposals), 'len: ', len(predicted_proposals))
        start = time.perf_counter()
        for el in predicted_proposals:
            # el = el.cpu()
            print('{}\t{}'.format(el.device, el.size() ) )
        print('\npredicted_objectness type: ', type(predicted_objectness), 'len: ', len(predicted_objectness))
        for el in predicted_objectness:
            # el = el.cpu()
            print('{}\t{}'.format(el.device, el.size() ) )
        print('\n')
        end = time.perf_counter()
        print('\t[proposals], device change time: ', end - start)

        start = time.perf_counter()
        
            # Find the top proposals by applying NMS and removing boxes that
            # are too small. The proposals are treated as fixed for approximate
            # joint training with roi heads. This approach ignores the derivative
            # w.r.t. the proposal boxes’ coordinates that are also network
            # responses, so is approximate.
        proposals = find_top_rpn_proposals(
            predicted_proposals,
            predicted_objectness,
            images,
            self.nms_thresh,
            self.pre_nms_topk[self.training],
            self.post_nms_topk[self.training],
            self.min_box_side_len,
            self.training,
        )
        end = time.perf_counter()
        print('\t[proposals] find_top_rpn_proposals time: ', end - start)

        return proposals