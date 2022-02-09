import sys
import os
import cv2
import torch
import argparse
import pafy
import time
import numpy as np
# add project root directory to python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_ROOT)

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

from engines.rcnn_tracker import RcnnTracker
from utils.track_visualizer import TrackVisualizer
from utils.mots_evaluation import file_lines_from_instances, parse_mots_seqmap, crop_overlapping_masks, result_image_from_objects


CONFIDENCE_THRESHOLD = 0.65

def setup_cfg():

    cfg = get_cfg()
    cfg.merge_from_file(PROJECT_ROOT + '/configs/mask_rcnn_R_101_FPN_3x.yaml')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    cfg.freeze()

    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="RCNN tracker test")
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--sequence", help="Path to video sequence")
    parser.add_argument("--youtube", help="Youtube video URL")
    parser.add_argument("--output", help="Path where output frames will be saved")
    parser.add_argument("--mots_evaluation", help='Path to MOTS seqmap')
    return parser


def visualize_tracks(frame, objects, video_visualizer):

    objects = objects.to(torch.device('cpu'))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    vis_frame = video_visualizer.draw_instance_predictions(frame, objects)

    # Converts Matplotlib RGB format to OpenCV BGR format
    vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
    return vis_frame


def get_images_from_dir(path):

    extensions = ['jpg', 'png', 'bmp']
    contents = os.listdir(path)
    contents.sort()
    image_contents = [imgpath for imgpath in contents if imgpath.split('.')[-1] in extensions]

    return image_contents


args = get_parser().parse_args()
config = setup_cfg()
metadata = MetadataCatalog.get("coco_2017_val")
video_visualizer = TrackVisualizer(metadata)
evaluation_output_path = 'output/evaluation_results'
kitti_mots_images = 'datasets/data_tracking_image_2/training/image_02'
if not os.path.isdir(evaluation_output_path):
    os.mkdir(evaluation_output_path)


if args.webcam:

    assert args.sequence is None, "Cannot have both webcam and sequence input"

    cam = cv2.VideoCapture(0)
    image_size = ( int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT) ), int(cam.get(cv2.CAP_PROP_FRAME_WIDTH) ) )
    tracker = tracker = RcnnTracker(config, image_size, metadata=metadata)

    while(1):

        _, frame = cam.read()
        current_objects = tracker.next_frame(frame)
        visualized_result = visualize_tracks(frame, current_objects, video_visualizer)
        visualized_result = cv2.resize(visualized_result, (1280, 960))
        cv2.imshow('tracking', visualized_result)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('l'):
            cv2.imwrite('frame' + str(tracker.frame_count) +'.jpg', visualized_result)

    cv2.destroyAllWindows()


elif args.sequence:

    if os.path.isdir(args.sequence):

        img_filenames = get_images_from_dir(args.sequence)
        sample_frame = cv2.imread(os.path.join(args.sequence, img_filenames[0]) )
        image_size = sample_frame.shape[:2]
        tracker = RcnnTracker(config, image_size, metadata=metadata)
        
        fpses = []
        for imgname in img_filenames:
            
            frame = cv2.imread(os.path.join(args.sequence, imgname) )
            torch.cuda.synchronize()
            start_time = time.time()
            current_objects = tracker.next_frame(frame)
            torch.cuda.synchronize()
            end_time = time.time()
            fps = 1/(end_time - start_time)
            fpses.append(fps)
            print('fps:', fps)
            visualized_result = visualize_tracks(frame, current_objects, video_visualizer)
            cv2.imshow('tracking', visualized_result)
            if args.output:
                if not os.path.isdir(args.output):
                    os.mkdir(args.output)
                cv2.imwrite(args.output + '/{:06d}'.format(tracker.frame_count) + '.png', visualized_result)
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
            elif key == ord('l'):
                cv2.imwrite('frame' + str(tracker.frame_count) +'.jpg', visualized_result)

        print('average fps:', np.sum(fpses)/len(fpses))
        cv2.destroyAllWindows()

    else:
        print('path not found: ', args.sequence)


elif args.youtube:

    video = pafy.new(args.youtube)
    best = video.getbest(preftype="mp4")

    capture = cv2.VideoCapture()
    capture.open(best.url)
    image_size = ( int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) ), int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) ) )
    tracker = tracker = RcnnTracker(config, image_size, metadata=metadata)

    while(1):

        _, frame = capture.read()
        current_objects = tracker.next_frame(frame)
        visualized_result = visualize_tracks(frame, current_objects, video_visualizer)
        visualized_result = cv2.resize(visualized_result, (1920, 1200))
        cv2.imshow('tracking', visualized_result)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('l'):
            cv2.imwrite('frame' + str(tracker.frame_count) +'.jpg', visualized_result)

    cv2.destroyAllWindows()


elif args.mots_evaluation:

    sequences, seq_lengths = parse_mots_seqmap(args.mots_evaluation)
    print('Running evaluation for sequences:')
    for s in sequences:
        print(s)

    for seq in sequences:
        
        sequence_path = os.path.join(kitti_mots_images, seq)
        img_filenames = get_images_from_dir(sequence_path)
        sample_frame = cv2.imread(os.path.join(sequence_path, img_filenames[0]) )
        image_size = sample_frame.shape[:2]
        tracker = RcnnTracker(config, image_size, metadata=metadata)
        # output_file_string = ''
        # output_file = evaluation_output_path + '/' + seq + '.txt'
        sequence_output_path = os.path.join(evaluation_output_path, seq)
        if not os.path.isdir(sequence_output_path):
            os.mkdir(sequence_output_path)
        print("\nEvaluating sequence: ", seq)

        for imgname in img_filenames:

            print(imgname, end='\r')
            frame = cv2.imread(os.path.join(sequence_path, imgname) )
            current_objects = tracker.next_frame(frame)
            crop_overlapping_masks(current_objects)
            result_img = result_image_from_objects(current_objects, image_size)
            result_imgname = '{:06d}'.format(tracker.frame_count-1) + '.png'
            cv2.imwrite(os.path.join(sequence_output_path, result_imgname), result_img)
            # output_file_string += file_lines_from_instances(current_objects, tracker.frame_count-1, image_size)
            # visualized_result = visualize_tracks(frame, current_objects, video_visualizer)
            # cv2.imshow('tracking', visualized_result)
            # key = cv2.waitKey(0)
            # if key == ord('q'):
            #     break


else:

    print("No input given, use --webcam, --youtube, --mots_evaluation or --sequence arguments")

