import os
import numpy as np
import torch
import cv2
import math

from detectron2.utils.colormap import random_color
from pycocotools.mask import toBbox, decode

from engines.roi_features_generator import RoiFeaturesGenerator
from utils.mots_evaluation import parse_mots_seqmap


def visualize_frame(frame_objects, frame):

    for instance in frame_objects:
        color = tuple( int(x) for x in random_color() )
        topleft_pt = (instance[2], instance[3])
        bottomright_pt = (instance[2] + instance[4], instance[3] + instance[5])
        frame = cv2.rectangle(frame, topleft_pt, bottomright_pt, color, 2)
    cv2.imshow('frame', frame)
    cv2.waitKey(0)


class MOTloader:

    """
        Training data loader for association_head
        loads MOT dataset, prepares ROI features from frames, packs these features into batches
    """

    def __init__(self, config, sequence_path, frames_in_batch=8, roi_size=8):

        self.config = config
        self.frames_in_batch = frames_in_batch
        self.sequence_path = sequence_path
        self.sequence_info = self.read_seqinfo()
        self.frames_in_sequence = int(self.sequence_info['seqLength'])
        self.num_of_batches = math.floor(self.frames_in_sequence/self.frames_in_batch)
        self.roi_generator = RoiFeaturesGenerator(config=self.config, roi_size=roi_size)
        #sequence_objects: nparray <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>
        #from whole sequence
        self.sequence_objects = self.gt_instances_from_sequence()


    def read_seqinfo(self):

        seqinfo_dict = {}
        seqinfo_path = self.sequence_path + '/seqinfo.ini'
        with open(seqinfo_path, 'r') as seqinfo_file:
            file_lines = seqinfo_file.readlines()
            contents = [line.split('=') for line in file_lines]
            for el in contents:
                if len(el) > 1:
                    seqinfo_dict[el[0]] = el[1].strip()

        return seqinfo_dict


    def gt_instances_from_sequence(self):

        gt_txt_path = self.sequence_path + '/gt/gt.txt'
        with open(gt_txt_path, 'r') as gt_file:
            gt_file_lines = gt_file.readlines()
            # [:7] - ignore 3d coordinates
            # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>
            gt_instances = [line.split(',')[:7] for line in gt_file_lines]
            gt_instances = np.array( [[int(el) for el in line] for line in gt_instances] )
            gt_instances = gt_instances[np.where(gt_instances[:, 6] == 1)]

        return np.array(gt_instances)


    def objects_from_frame(self, frame_number):

        return self.sequence_objects[np.where(self.sequence_objects[:, 0] == frame_number)]


    def frame_from_sequence(self, frame_number):

        return cv2.imread(self.sequence_path + '/img1/{:06d}.jpg'.format(frame_number))


    def rois_ids_from_frame(self, frame_number):

        frame = self.frame_from_sequence(frame_number)
        frame_objects = self.objects_from_frame(frame_number)
        ids, rois = self.roi_generator.get_rois_features(frame, frame_objects)

        return ids, rois

    def get_training_batch(self, batch_idx):

        """
            returns:
                np.array, torch.Tensor
                where np array is a vector of ids of shape (N,) and
                tensor is a feature batch of shape (N, C, ROI_SIZE, ROI_SIZE)
        """

        assert batch_idx < self.num_of_batches

        batch_ids = []
        batch_rois = []

        for frame_idx_in_batch in range(self.frames_in_batch):

            #explicit frame number in dataset
            frame_number = (frame_idx_in_batch + 1) + batch_idx * self.frames_in_batch
            frame = self.frame_from_sequence(frame_number)
            frame_objects = self.objects_from_frame(frame_number)
            ids, rois = self.roi_generator.get_rois_features(frame, frame_objects)
            batch_ids.append(ids)
            batch_rois.append(rois)

        batch_ids = torch.cat(batch_ids)
        batch_rois = torch.cat(batch_rois)

        return batch_ids, batch_rois



class MOTSloader:

    """
        Training data loader for association_head
        loads MOTS dataset, prepares ROI features from frames, packs these features into batches
    """

    def __init__(self, config, dataset_path, seqmap_path, frames_in_batch=8, roi_size=8):

        self.config = config
        self.frames_in_batch = frames_in_batch
        self.dataset_path = dataset_path
        self.seqmap_names, self.seqmap_lengths = parse_mots_seqmap(seqmap_path)
        self.num_of_sequences = len(self.seqmap_names)
        self.roi_generator = RoiFeaturesGenerator(config=self.config, roi_size=roi_size)
        #sequence_objects: dict: seqname > nparray <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>
        self.sequence_objects = self.gt_instances_from_sequence()
        print('reading instances done')
        self.frames_with_objects_per_seq = self.find_frames_with_objects()
        self.batches_per_sequence = [math.floor( len(self.frames_with_objects_per_seq[seq]) / self.frames_in_batch ) for seq in self.seqmap_names]
        self.num_of_batches = np.array(self.batches_per_sequence).sum()
        # print('frames_with_objects:', self.frames_with_objects_per_seq)
        # print('batches_per_seq:', self.batches_per_sequence)
        ##show dataset with bboxes
        # for seq_idx, seq in enumerate(self.seqmap_names):
        #     print(seq)
        #     for frame_num in self.frames_with_objects_per_seq[seq]:
        #         print(frame_num)
        #         visualize_frame(self.objects_from_frame(seq, frame_num), self.frame_from_sequence(seq, frame_num))


    def find_frames_with_objects(self):

        result_dict = {}
        for seq_idx, seq in enumerate(self.seqmap_names):
            result_dict[seq] = np.unique(self.sequence_objects[seq][0][:, 0])

        return result_dict


    def gt_instances_from_sequence(self):

        '''
        returns:
        dict[Str: seqname] -> tuple(np.array, pycocotools mask dict)
        '''

        instances_txt_path = self.dataset_path + '/instances_txt/'
        objects_dict = {}

        for seqname in self.seqmap_names:
            with open(instances_txt_path + seqname + '.txt', 'r') as gt_file:
                gt_file_lines = gt_file.readlines()
                # 52 1005 1 375 1242 WSV:2d;1O10000O10000O1O100O100O1O100O1000000000000000O100O102N5K00O1O1N2O110OO2O001O1NTga3
                # Which means
                # time frame 52
                # object id 1005 (meaning class id is 1, i.e. car and instance id is 5)
                # class id 1
                # image height 375
                # image width 1242
                # rle WSV:2d;1O10000O10000O1O100O100O1O100O1000000000000000O100O...1O1N 
                seq_objects = []
                seq_masks = []
                for line in gt_file_lines:
                    obj_info = line.split(' ')
                    frame_num = int(obj_info[0])
                    ob_id = int(obj_info[1])
                    height = int(obj_info[3])
                    width = int(obj_info[4])
                    rle = obj_info[5].strip()
                    mask = {'size': [height, width], 'counts': rle}
                    # if ob_id == 10000:
                    #     bin_mask = decode(mask)
                    #     print(frame_num)
                    #     cv2.imshow('co to', bin_mask*255)
                    #     cv2.waitKey(0)
                    bbox = [int(coord) for coord in toBbox(mask)]
                    if ob_id != 10000:
                        # bin_mask = torch.tensor(decode(mask))
                        seq_masks.append(mask)
                        seq_objects.append([ frame_num, ob_id, bbox[0], bbox[1], bbox[2], bbox[3] ])
            
            objects_dict[seqname] = (np.array(seq_objects), seq_masks)

        return objects_dict


    def objects_masks_from_frame(self, sequence_name, frame_number):

        objects_from_sequence = self.sequence_objects[sequence_name][0]
        masks_from_sequence = self.sequence_objects[sequence_name][1]
        # print('objects from sequence:', objects_from_sequence)
        indices_from_frame = np.where(objects_from_sequence[:, 0] == frame_number)
        frame_objects = objects_from_sequence[ indices_from_frame ]
        frame_masks = [ masks_from_sequence[i] for i in indices_from_frame[0].tolist() ]
        # print('frame_masks:', len(frame_masks), len(frame_objects))
        # print('frame', frame_number, 'objects:', frame_objects)
        return frame_objects, frame_masks


    def frame_from_sequence(self, sequence_name, frame_number):

        frame_path = self.dataset_path + '/training/image_02/' + sequence_name + '/{:06d}.png'.format(frame_number)
        return cv2.imread(frame_path)


    def rois_ids_from_frame(self, sequence_name, frame_number):

        frame = self.frame_from_sequence(sequence_name, frame_number)
        frame_objects, frame_masks = self.objects_masks_from_frame(sequence_name, frame_number)
        ids, rois = self.roi_generator.get_rois_features(frame, frame_objects)

        return ids, rois


    def get_training_batch(self, sequence_idx, batch_idx):

        """
            returns:
                torch.Tensor, torch.Tensor
                vector of ids of shape (N,) and
                tensor is a feature batch of shape (N, C, ROI_SIZE, ROI_SIZE)
        """

        assert batch_idx < self.batches_per_sequence[sequence_idx]
        assert sequence_idx < self.num_of_sequences

        seqname = self.seqmap_names[sequence_idx]
        batch_ids = []
        batch_rois = []

        ## create batch of objects from 'self.frames_in_batch' consecutive frames with objects
        for frame_idx_in_batch in range(self.frames_in_batch):

            frame_number = self.frames_with_objects_per_seq[seqname][frame_idx_in_batch + batch_idx * self.frames_in_batch]
            ids, rois = self.rois_ids_from_frame(seqname, frame_number)
            batch_ids.append(ids)
            batch_rois.append(rois)

        batch_ids = torch.cat(batch_ids)
        batch_rois = torch.cat(batch_rois)

        return batch_ids, batch_rois