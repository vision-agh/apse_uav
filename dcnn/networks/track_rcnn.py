
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN


class TrackRCNN(GeneralizedRCNN):

    """
    rcnn architecture derived from mask-rcnn which outputs also backbone features
    during inference
    """

    def __init__(self, cfg):
        super().__init__(cfg)


    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
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

        images = self.preprocess_image(batched_inputs)
        # print('image tensor: ', images.tensor.size())
        # print('image tensor: ', images.tensor[0, 0, :, :])
        # print('backbone weights:')
        # for i, param in enumerate( self.backbone.parameters() ):
        #     if i < 3:
        #         print(param.data)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes), features
        else:
            return results, features