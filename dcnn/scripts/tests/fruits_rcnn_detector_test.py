import sys
import os
import cv2
import torch
# add project root directory to python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_ROOT)

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.data.datasets import register_coco_instances, load_coco_json


CONFIDENCE_THRESHOLD = 0.5
DATASET_JSON = PROJECT_ROOT + "/datasets/data/trainval.json"
DATASET_IMAGES = PROJECT_ROOT + "/datasets/data/images"

def setup_cfg():

    cfg = get_cfg()
    cfg.merge_from_file(PROJECT_ROOT + '/configs/mask_rcnn_R_50_FPN_3x.yaml')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    cfg.MODEL.WEIGHTS = PROJECT_ROOT + "/output/model_final.pth"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.freeze()

    return cfg


def visualize_detections(frame, detections, video_visualizer):

    detections = detections.to(torch.device('cpu'))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    vis_frame = video_visualizer.draw_instance_predictions(frame, detections)

    # Converts Matplotlib RGB format to OpenCV BGR format
    vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
    return vis_frame


cfg = setup_cfg()
register_coco_instances("fruits_nuts", {}, DATASET_JSON, DATASET_IMAGES)
MetadataCatalog.get("fruits_nuts").thing_classes = ['date', 'fig', 'hazelnut']
metadata = MetadataCatalog.get('fruits_nuts')
print(metadata)
predictor = DefaultPredictor(cfg)
video_visualizer = VideoVisualizer(metadata)

imgname = sys.argv[1]
imgpath = PROJECT_ROOT + '/test_images/' + imgname
image = cv2.imread(imgpath)
predictions = predictor(image)
predictions = predictions['instances']

visualized_result = visualize_detections(image, predictions, video_visualizer)
cv2.imshow('detekcje ' + imgname, visualized_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
