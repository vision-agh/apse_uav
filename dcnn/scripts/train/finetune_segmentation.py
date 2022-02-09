import os
import sys
import torch
from torch import optim
import numpy as np
import cv2
import random
import tqdm
from sklearn.model_selection import KFold
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

# np.set_printoptions(threshold=sys.maxsize)
# add project root directory to python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_ROOT)

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.evaluation import inference_context
from detectron2.utils.video_visualizer import VideoVisualizer

from utils.UAV_utils import build_detection_test_loader, build_detection_train_loader, generate_uav_dataset_dictionaries, draw_bboxes
from utils.COCO_utils import detectron2_dataset_to_coco, generate_coco_dataset_dictionaries


COCO_CATEGORY_IDS_TO_UAV = {
    1: 3,   # coco person
    3: 0,   # coco car
    6: 2,   # coco bus
    8: 1    # coco truck
}
ORIGINAL_FASTER_WEIGHTS_PATH = './pretrained/R_50_FPN_UAV_700.pth'
RESUME = True
START_CHECKPOINT_PATH = './output/R_50_FPN_UAV_SEGM/R_50_FPN_UAV_SEGM_last.pth'
CHECKPOINT_NAME = 'R_50_FPN_UAV_SEGM'
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output', CHECKPOINT_NAME)
COCO_DATASET_PATH = './datasets/coco_train2017'
COCO_JSON = os.path.join(COCO_DATASET_PATH, 'annotations/instances_train2017.json')
COCO_IMGFOLDER = os.path.join(COCO_DATASET_PATH, 'train2017')
RESULTS_FILE = 'results.txt'
NUM_CLASSES = 4
K_FOLDS = 1000    # ignored if kfold data available in checkpoint SPLITS ONLY UAV DATASET
CLASSES_NAMES = ['car', 'truck', 'bus', 'person']
MAX_ITER = 20000
CHECKPOINT_PERIOD = 10


def merge_full_mask_rcnn(original_weights_path, segmentation_model):

    faster_checkpoint = torch.load(original_weights_path)['model']
    result_state_dict = segmentation_model.state_dict()
    for k, v in faster_checkpoint.items():
        if k not in segmentation_model.state_dict():
            result_state_dict[k] = v

    return result_state_dict


def show_dataset(dataset_dicts):

    for img_dict in dataset_dicts:

        img = cv2.imread(img_dict['file_name'])
        bboxes = [ann['bbox'] for ann in img_dict['annotations']]
        classes = [ann['category_id'] for ann in img_dict['annotations']]
        ids = [ann['target_id'] for ann in img_dict['annotations']]
        gt_img = draw_bboxes(img, bboxes, classes=classes, ids=ids)
        cv2.imshow('test', gt_img)
        print('\nFilename:', img_dict['file_name']) 
        print('Annotations:')
        for el in img_dict['annotations']:
            print(el)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def setup_cfg():

    cfg = get_cfg()
    cfg.merge_from_file(PROJECT_ROOT + '/configs/mask_rcnn_R_50_FPN_3x.yaml')
    # cfg.MODEL.WEIGHTS = "./pretrained/mask_rcnn_R_50_FPN_3x_original/model_700.pth"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.MASK_ON = True
    cfg.MODEL.LOAD_PROPOSALS = True
    cfg.MODEL.PROPOSAL_GENERATOR.NAME = 'PrecomputedProposals'
    cfg.freeze()

    return cfg


def visualize_detections(frame, detections, video_visualizer):

    detections = detections.to(torch.device('cpu'))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    vis_frame = video_visualizer.draw_instance_predictions(frame, detections)

    # Converts Matplotlib RGB format to OpenCV BGR format
    vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
    return vis_frame


def do_test(model, test_dataloader, visualize=False):

    with inference_context(model), torch.no_grad():

        test_dataset_results = []
        test_dataset_gt = []
        idx_to_visualize = random.randint(0, len(test_dataloader))
        # pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader)) 
        for idx, inputs in enumerate(test_dataloader):
            # print('inputs:', len(inputs))
            print('Testing... {}/{}'.format(idx, len(test_dataloader)), end='\r')
            outputs = model(inputs)

            if visualize and idx_to_visualize == idx:
                video_visualizer = VideoVisualizer({'thing_classes': CLASSES_NAMES})
                imgpath = inputs[0]['file_name']
                frame = cv2.imread(imgpath)
                visualized_result = visualize_detections(frame, outputs[0]['instances'], video_visualizer)
                gt_bboxes = [ann['bbox'] for ann in inputs[0]['annotations']]
                visualized_result = draw_bboxes(visualized_result, gt_bboxes)
                
                cv2.imwrite(os.path.join(OUTPUT_DIR, 'test_results.png'), visualized_result)
                # cv2.imshow(str(inputs[0]['image_id']), visualized_result)
            
            test_dataset_gt += inputs   # list of images (batch)
            image_ids = [img_dict['image_id'] for img_dict in inputs]
            coco_results = [instances_to_coco_json(result['instances'].to('cpu'), image_id) for result, image_id in zip(outputs, image_ids)]
            coco_results = [instance for instances_per_img in coco_results for instance in instances_per_img]   #flatten list of batched outputs
            test_dataset_results += coco_results    # list of detections for batch

        if len(test_dataset_results) > 0 and len(test_dataset_gt) > 0:
            coco_gt = detectron2_dataset_to_coco(test_dataset_gt, CLASSES_NAMES)
            coco_dt = coco_gt.loadRes(test_dataset_results)

            coco_evaluator = COCOeval(cocoGt=coco_gt, cocoDt=coco_dt, iouType='segm')
            coco_evaluator.evaluate()
            coco_evaluator.accumulate()
            coco_evaluator.summarize()

        return coco_evaluator.stats


if not os.path.isdir(os.path.join(OUTPUT_DIR, 'visualized')):
    print('Creating output path:', OUTPUT_DIR)
    os.makedirs(os.path.join(OUTPUT_DIR, 'visualized'))            

cfg = setup_cfg()
model = build_model(cfg)
parameters = [p for name, p in model.named_parameters() if 'roi_heads.mask_head' in name]
optimizer = optim.SGD(parameters, lr=0.02, momentum=0.9)
scheduler = build_lr_scheduler(cfg, optimizer)
coco_dataset_dicts = generate_coco_dataset_dictionaries(COCO_JSON, COCO_IMGFOLDER,
                                                        allowed_classes=CLASSES_NAMES,
                                                        category_mapping=COCO_CATEGORY_IDS_TO_UAV,
                                                        precomputed_proposals=True)
# show_dataset(coco_dataset_dicts)

if RESUME:
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(START_CHECKPOINT_PATH)
    chkpt = torch.load(START_CHECKPOINT_PATH)
    print('Resume training from checkpoint ' + START_CHECKPOINT_PATH)
    for k, v in chkpt.items():
        print(k, ":", type(v))
    start_iter = chkpt['iteration'] + 1
    kfold_split = chkpt['kfold_split']
    best_precision = chkpt['best_precision']
    best_recall = chkpt['best_recall']
    K_FOLDS = chkpt['k_folds']

    optimizer.load_state_dict(chkpt['optimizer'])
    scheduler.load_state_dict(chkpt['scheduler'])

    with open(os.path.join(OUTPUT_DIR, RESULTS_FILE), 'w') as file:
        file.write(chkpt['training_results'])  

else:
    checkpointer = DetectionCheckpointer(model)
    chkpt = checkpointer.load(START_CHECKPOINT_PATH)
    print('Start training with checkpoint ' + START_CHECKPOINT_PATH)
    start_iter = 0
    kfold = KFold(n_splits=K_FOLDS, shuffle=True)
    kfold_split = list(kfold.split(coco_dataset_dicts))
    kfold_split = kfold_split[0]    #CROSS VALIDATION not implemented yet, take first fold
    best_precision = 0
    best_recall = 0
    with open(os.path.join(OUTPUT_DIR, RESULTS_FILE), 'w') as file:
        file.write('\t\tAP\tAP_05\tAP0.75\tAP_s\tAP_m\tAP_l\tAR_1\tAR_10\tAR_100\tAR_s\tAR_m\tAR_l\n')  


(train_ids, test_ids) = kfold_split
train_coco_dataset_dicts = [coco_dataset_dicts[i] for i in train_ids]
test_coco_dataset_dicts = [coco_dataset_dicts[i] for i in test_ids]
train_dataloader = build_detection_train_loader(cfg, train_coco_dataset_dicts)
test_dataloader = build_detection_test_loader(cfg, test_coco_dataset_dicts)
print('Loaded {} coco images for training'.format(len(coco_dataset_dicts)))

model.train()
print('Training for {} iterations started'.format(MAX_ITER))
with EventStorage(0) as storage:
    for data, iteration in zip(train_dataloader, range(start_iter, MAX_ITER)):

        # print('batch filenames:')
        # batch_filenames = [img_dict['file_name'] for img_dict in data]
        # for el in batch_filenames:
        #     print(el)

        loss_dict = model(data)
        losses = sum(loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        scheduler.step()

        print('Iter. {}/{}:\tloss: {}'.format(iteration, MAX_ITER, losses))
        if iteration != 0 and iteration % CHECKPOINT_PERIOD == 0:

            test_results = do_test(model, test_dataloader, visualize=True)
            print('test results:', test_results)
            if test_results is not None:
                results_string = ['{:.3f}'.format(el) for el in test_results]
                results_string = '{}/{}:\t'.format(iteration, MAX_ITER) + '\t'.join(results_string)
                AP_all = test_results[0]
                AR_all = test_results[8]
                if AP_all > best_precision:
                    best_precision = AP_all
                    results_string += ' Best precision!'
                if AR_all > best_recall:
                    best_recall = AR_all
                    results_string += ' Best recall!'
                with open(os.path.join(OUTPUT_DIR, RESULTS_FILE), 'a') as file:
                    file.write(results_string + '\n')

            with open(os.path.join(OUTPUT_DIR, RESULTS_FILE), 'r') as file:
                checkpoint_dict = {
                    'model': merge_full_mask_rcnn(ORIGINAL_FASTER_WEIGHTS_PATH, model),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': iteration,
                    'kfold_split': kfold_split,
                    'k_folds': K_FOLDS,
                    'best_precision': best_precision,
                    'best_recall': best_recall,
                    'training_results': file.read()
                }

            last_chkpt = CHECKPOINT_NAME + "_last"
            torch.save(checkpoint_dict, os.path.join(OUTPUT_DIR, last_chkpt+'.pth'))

            if test_results is not None:
                if AP_all == best_precision:
                    best_AP_chkpt = CHECKPOINT_NAME + "_bestAP"
                    torch.save(checkpoint_dict, os.path.join(OUTPUT_DIR, best_AP_chkpt+'.pth'))

                if AR_all == best_recall:
                    best_AR_chkpt = CHECKPOINT_NAME + "_bestAR"
                    torch.save(checkpoint_dict, os.path.join(OUTPUT_DIR, best_AR_chkpt+'.pth'))
            
            del checkpoint_dict