import cv2
import os
import numpy as np
import torch
import math
import random
import re
import time
import logging
import operator
import copy
from torch.utils.data import Dataset
from PIL import Image

from detectron2.structures.boxes import Boxes, BoxMode
from detectron2.structures.instances import Instances
import detectron2.data.transforms as T
from detectron2.data.build import worker_init_reset_seed
from detectron2.data import samplers
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.data.common import AspectRatioGroupedDataset, DatasetFromList, MapDataset
# from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.detection_utils import check_metadata_consistency
from detectron2.utils.comm import get_world_size
import detectron2.data.detection_utils as utils


IGNORED_SEQUENCES = ['M0601', 'M0207']
IGNORED_IDS_PER_SEQUENCE = {
    'M0606': [68, 69, 112, 71],
    'M0501': [10],
    'M1007': [36],
    'M0210': [34, 6, 2, 11, 10, 30, 18],
    'M1401': [42, 66],
    'M1304': [75, 76],
    'M0204': [22],
    'M1001': [11],
    'M0802': [23, 53]
}


def fabricate_outputs(gt_img_dict):

    results = []

    for ann in gt_img_dict['annotations']:

        result = {
            'image_id': gt_img_dict['image_id'],
            'category_id': ann['category_id'],
            'bbox': ann['bbox'],
            'score': 1
        }

        results.append(result)

    return results

    


def get_images_from_dir(path):

    extensions = ['jpg', 'png', 'bmp']
    contents = os.listdir(path)
    contents.sort()
    image_contents = [imgpath for imgpath in contents if imgpath.split('.')[-1] in extensions and 'Annotated' not in imgpath]

    return image_contents


def draw_bboxes(img, boxes, classes=None, ids=None):

    text_color = (0, 0, 255) # BGR
    font_scale = 1
    thickness = 1
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    result = img

    for idx, box in enumerate(boxes):
        start_point = (int(box[0]), int(box[1]))
        end_point = (int(box[0]) + int(box[2]), int(box[1]) + int(box[3]))
        result = cv2.rectangle(img, start_point, end_point, (0, 255, 0), 1)
        if classes is not None:
            text_pos = (int(box[0]), int(box[1]) + 15)   # up-left corner
            result = cv2.putText(result, str(classes[idx]), text_pos, font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA) 
        if ids is not None:
            text_pos = (int(box[0]), int(box[1]) + int(box[3]) - 5) #down-left corner
            result = cv2.putText(result, str(ids[idx]), text_pos, font, font_scale, text_color, thickness, cv2.LINE_AA) 

    return result


def get_instances_from_frame(imgname, instances, seqname, ignored_ids=None, category_mapping=None):

    frame_number = int(imgname.strip('img').strip('.jpg'))
    frame_indices = np.where(instances[:, 0] == frame_number)
    frame_objects = instances[frame_indices]
    frame_boxes = frame_objects[:, 2:6]
    frame_classes = frame_objects[:, 8]

    annotations = []
    for instance in frame_objects:
        target_id = instance[1]
        if ignored_ids is None or target_id not in ignored_ids:
            bbox = list(instance[2:6])
            obj_class = instance[8] - 1
            if category_mapping:
                obj_class = category_mapping[obj_class]
            occlusion = instance[7]
            if occlusion != 2:   # ignore large occlusions
                obj_dict = {'is_crowd': 0,
                            'bbox': bbox,
                            'category_id': int(obj_class),
                            'bbox_mode': BoxMode.XYWH_ABS,
                            'target_id': int( seqname[1:] + imgname.strip('img').strip('.jpg') + str(target_id) )} # create unique target id among ALL datasets
                annotations.append(obj_dict)

    return annotations


def get_instances_from_sequence(gt_filepath):

    with open(gt_filepath, "r") as gt_file:
        filelines = gt_file.readlines()
        gt_instances = [line.split(',') for line in filelines]
        gt_instances = np.array( [[int(el) for el in line] for line in gt_instances] )

    return gt_instances


def generate_uav_dataset_dictionaries(dataset_dir, allowed=None, category_mapping=None):

    result = []
    sequences = os.listdir(dataset_dir)
    print('IGNORED:', IGNORED_SEQUENCES)
    if allowed:
        sequences = [path for path in sequences if path in allowed and 'GT' not in path and 'attr' not in path and path not in IGNORED_SEQUENCES]
    else:
        sequences = [path for path in sequences if 'GT' not in path and 'attr' not in path and path not in IGNORED_SEQUENCES]

    print('Loading sequences {}:'.format(len(sequences)))
    for seq in sequences:
        print(os.path.join(dataset_dir, seq))
        seq_gt_filepath = os.path.join(dataset_dir, 'GT', seq + '_gt_whole.txt')
        seq_instances = get_instances_from_sequence(seq_gt_filepath)
        imgnames = get_images_from_dir(os.path.join(dataset_dir, seq))
        
        for imgname in imgnames:
            im_path = os.path.join(dataset_dir, seq, imgname)
            im_pil = Image.open(im_path)
            width, height = im_pil.size
            annotations = get_instances_from_frame(imgname, seq_instances, seq, ignored_ids=IGNORED_IDS_PER_SEQUENCE.get(seq), category_mapping=category_mapping)
            imgdict = {'file_name': im_path,
                        'height': height,
                        'width': width,
                        'image_id': int(imgname.strip('img').strip('.jpg')),
                        'annotations': annotations}
            result.append(imgdict)

    return result


def build_detection_train_loader(cfg, dataset_dicts, mapper=None):
    """
    A data loader is created by the following steps:

    1. Use the dataset names in config to query :class:`DatasetCatalog`, and obtain a list of dicts.
    2. Start workers to work on the dicts. Each worker will:

       * Map each metadata dict into another format to be consumed by the model.
       * Batch them by simply putting dicts into a list.

    The batched ``list[mapped_dict]`` is what this dataloader will return.

    Args:
        cfg (CfgNode): the config
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            By default it will be `DatasetMapper(cfg, True)`.

    Returns:
        an infinite iterator of training data
    """
    num_workers = get_world_size()
    images_per_batch = cfg.SOLVER.IMS_PER_BATCH
    assert (
        images_per_batch % num_workers == 0
    ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of workers ({}).".format(
        images_per_batch, num_workers
    )
    assert (
        images_per_batch >= num_workers
    ), "SOLVER.IMS_PER_BATCH ({}) must be larger than the number of workers ({}).".format(
        images_per_batch, num_workers
    )
    images_per_worker = images_per_batch // num_workers

    # print('dataset_dicts:', type(dataset_dicts))
    # for img_dict in dataset_dicts:
    #     print()
    #     for k, v in img_dict.items():
    #         if k == 'annotations':
    #             print(k, ":", len(v))
    #             for el in v:
    #                 print()
    #                 for k, v in el.items():
    #                     print(k, ":", v, type(v))
                    
    #         else:
    #             print(k, ":", v)

        #     else:
        #         print(k, ":", v)
        #         boxes = [instance['bbox'] for instance in img_dict['annotations']]
        #         img = cv2.imread(v)
        #         vis_img = draw_bboxes(img, boxes)
        #         cv2.imshow(v, vis_img)

        # if cv2.waitKey(0) == ord('q'):
        #     break

    dataset = DatasetFromList(dataset_dicts, copy=False)

    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    dataset = MapDataset(dataset, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    if sampler_name == "TrainingSampler":
        sampler = samplers.TrainingSampler(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        sampler = samplers.RepeatFactorTrainingSampler(
            dataset_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
        )
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))
    
    if cfg.DATALOADER.ASPECT_RATIO_GROUPING:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=None,
            collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        data_loader = AspectRatioGroupedDataset(data_loader, images_per_worker)
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_worker, drop_last=True
        )
        # drop_last so the batch always have the same size
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=batch_sampler,
            collate_fn=trivial_batch_collator,
            worker_init_fn=worker_init_reset_seed,
        )

    return data_loader


def build_detection_test_loader(cfg, dataset_dicts, mapper=None):
    """
    Similar to `build_detection_train_loader`.
    But this function uses the given `dataset_name` argument (instead of the names in cfg),
    and uses batch size 1.

    Args:
        cfg: a detectron2 CfgNode
        dataset_name (str): a name of the dataset that's available in the DatasetCatalog
        mapper (callable): a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           By default it will be `DatasetMapper(cfg, False)`.

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.
    """

    dataset = DatasetFromList(dataset_dicts)
    if mapper is None:
        mapper = DatasetMapper(cfg, False)
    dataset = MapDataset(dataset, mapper)

    sampler = samplers.InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


class DatasetMapper:
    """
    !!!Modified so it does not remove 'annotations' field during evaluation!!!

    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger(__name__).info("CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None

        self.tfm_gens = utils.build_transform_gen(cfg, is_train)
        self.tfm_gens.append(T.RandomBrightness(intensity_min=0.9, intensity_max=1.1))
        self.tfm_gens.append(T.RandomSaturation(intensity_min=0.9, intensity_max=1.1))
        self.tfm_gens.append(T.RandomContrast(intensity_min=0.9, intensity_max=1.1))
        self.tfm_gens.append(T.RandomLighting(scale=0.2))
        print('Transform used in data augmentation:', str(self.tfm_gens))


        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = cfg.MODEL.MASK_ON
        self.mask_format    = cfg.INPUT.MASK_FORMAT
        self.keypoint_on    = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        # fmt: on
        if self.keypoint_on and is_train:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)
        else:
            self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.min_box_side_len = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if "annotations" not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # USER: Remove if you don't use pre-computed proposals.
        if self.load_proposals:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, self.min_box_side_len, self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            # dataset_dict.pop("annotations", None)
            # dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )
            # Create a tight bounding box from masks, useful when image is cropped
            if self.crop_gen and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            with PathManager.open(dataset_dict.pop("sem_seg_file_name"), "rb") as f:
                sem_seg_gt = Image.open(f)
                sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")
            sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            dataset_dict["sem_seg"] = sem_seg_gt
        return dataset_dict


class UAVDataloader():

    def __init__(self, config, DATASET_DIR, imgs_per_batch = 2, randomize = True, oneclass = False):

        self.randomize = randomize
        self.oneclass = oneclass
        self.config = config
        self.DATASET_DIR = DATASET_DIR
        self.imgs_per_batch = imgs_per_batch
        self.sequences = self.get_birdview_sequences()
        # self.sequences = ['M0203_test', 'M0703_bird', 'M0801_test']
        self.instances_per_sequence = {}
        self.imgnames_per_sequence = {}
        self.num_batches = 0
        for seq in self.sequences:
            self.instances_per_sequence[seq] = self.get_instances_from_sequence(seq)
            self.imgnames_per_sequence[seq] = get_images_from_dir(os.path.join(self.DATASET_DIR, seq))
            self.num_batches += len(self.imgnames_per_sequence[seq])


    def __iter__(self):

        self.iter_sequence_cnt = 0
        self.iter_frame_in_sequence_cnt = 0
        return self


    def __next__(self):

        if self.randomize:

            result = []
            for i in range(self.imgs_per_batch):
                seqindex = random.randrange(0, len(self.sequences))
                seqname = self.sequences[seqindex]
                frameindex = random.randrange(0, len(self.imgnames_per_sequence[seqname]))
                imgname = self.imgnames_per_sequence[seqname][frameindex]
                result += self.get_instances_from_frame(seqname, imgname)
            return result

        else:

            if self.iter_sequence_cnt < len(self.sequences):
                current_seqname = self.sequences[self.iter_sequence_cnt]
                if self.iter_frame_in_sequence_cnt < len(self.imgnames_per_sequence[current_seqname] ) - (self.imgs_per_batch - 1):
                    result = []
                    for i in range(self.imgs_per_batch):
                        result += self.get_instances_from_frame(current_seqname, self.imgnames_per_sequence[current_seqname][self.iter_frame_in_sequence_cnt])
                        self.iter_frame_in_sequence_cnt += 1
                    return result
                else:
                    self.iter_frame_in_sequence_cnt = 0
                    self.iter_sequence_cnt += 1
                    return self.__next__()
            else:
                raise StopIteration


    def get_birdview_sequences(self):

        contents = os.listdir(self.DATASET_DIR)
        result = [path for path in contents if 'GT' not in path and 'attr' not in path]

        return result


    def get_instances_from_sequence(self, seqname):

        seqname = seqname.split('_')[0]
        gt_filepath = os.path.join(self.DATASET_DIR, 'GT', seqname + '_gt_whole.txt')
        with open(gt_filepath, "r") as gt_file:
            filelines = gt_file.readlines()
            gt_instances = [line.split(',') for line in filelines]
            gt_instances = np.array( [[int(el) for el in line] for line in gt_instances] )
        
        return gt_instances


    def get_instances_from_frame(self, seqname, imgname):

        frame_number = int(imgname.strip('img').strip('.jpg'))
        frame_indices = np.where(self.instances_per_sequence[seqname][:, 0] == frame_number)
        frame_objects = self.instances_per_sequence[seqname][frame_indices]
        frame_boxes = frame_objects[:, 2:6]
        frame_boxes = np.array([ [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]] for bbox in frame_boxes ] ).astype(float)
        # frame_boxes = Boxes(frame_boxes)
        frame_img = cv2.imread(os.path.join(self.DATASET_DIR, seqname, imgname))
        frame_instances = Instances( (frame_img.shape[0], frame_img.shape[1]) )
        if self.oneclass:
            gt_classes = torch.Tensor([0] * len(frame_objects)).long()
        else:
            gt_classes = torch.Tensor(frame_objects[:, 8] - 1).long()
        frame_instances._fields['gt_boxes'] = frame_boxes
        frame_instances._fields['gt_classes'] = gt_classes
        image_batch = self.preprocess_image(frame_img, frame_instances)
        image_batch[0]['file_name'] = os.path.join(self.DATASET_DIR, seqname, imgname)

        return image_batch


    # preprocess image and bboxes in 'instances'
    def preprocess_image(self, original_image, instances):

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
            result = transform.apply_box(instances.gt_boxes)
            result = Boxes( torch.Tensor(result) )
            instances.gt_boxes = result

            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width, "instances": instances}
            batched_inputs = [inputs]

            return batched_inputs