import sys
import os
import cv2
import torch
# add project root directory to python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_ROOT)

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.engine.defaults import DefaultPredictor

CONFIDENCE_THRESHOLD = 0.5

def setup_cfg():

    cfg = get_cfg()
    cfg.merge_from_file(PROJECT_ROOT + '/configs/mask_rcnn_R_50_FPN_3x.yaml')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    cfg.MODEL.WEIGHTS = "./pretrained/mask_rcnn_R50_FPN_aerial/model.pth"
    cfg.MODEL.MASK_ON = True
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


def get_images_from_dir(path):

    extensions = ['jpg', 'png', 'bmp']
    contents = os.listdir(path)
    contents.sort()
    image_contents = [imgpath for imgpath in contents if imgpath.split('.')[-1] in extensions and 'Annotated' not in imgpath]

    return image_contents


cfg = setup_cfg()
metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused")
predictor = DefaultPredictor(cfg)
video_visualizer = VideoVisualizer(metadata)

imgdir = './test_images'
for imgname in get_images_from_dir(imgdir):
    imgpath = os.path.join(imgdir, imgname)
    image = cv2.imread(imgpath)
    predictions = predictor(image)
    predictions = predictions['instances']
    visualized_result = visualize_detections(image, predictions, video_visualizer)
    # cv2.imwrite('detekcje.png', visualized_result)
    cv2.imshow('detekcje', visualized_result)
    key = cv2.waitKey(0)
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite('./output.png', visualized_result)


cv2.destroyAllWindows()
