import math
import numpy as np
from enum import IntEnum, unique
from typing import Iterator, List, Tuple, Union
import torch

from detectron2.structures.boxes import Boxes

class SetBoxes(Boxes):

    def __init__(self, tensor: torch.Tensor):
        super().__init__(tensor)

    def __setitem__(self, item: Union[int, slice, torch.BoolTensor], value: Boxes) -> "Boxes":
        """substitute some of the boxes with given boxes"""
        self.tensor[item] = value.tensor

    @classmethod
    def cat(cls, boxes_list: List["Boxes"]) -> "Boxes":
        return SetBoxes(super().cat(boxes_list).tensor)