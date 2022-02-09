from collections import OrderedDict

from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer


class PartialCheckpointer(DetectionCheckpointer):

    def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
        super().__init__(model)

    def _load_model(self, checkpoint):

        new_state_dict = OrderedDict()
        for key, value in checkpoint["model"].items():
            new_key = key.split('backbone.')[-1]
            new_state_dict[new_key] = value
        checkpoint["model"] = new_state_dict

        incompatible = super()._load_model(checkpoint)
        return incompatible
    