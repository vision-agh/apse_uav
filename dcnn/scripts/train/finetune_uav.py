import os
import sys
import torch
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
from utils.COCO_utils import detectron2_dataset_to_coco, generate_coco_dataset_dictionaries, COCO_CATEGORY_IDS_TO_UAV
from utils.visdrone_utils import generate_visdrone_dataset_dictionaries
from utils.utils import plot_training_results, build_finetune_optimizer



CHECKPOINT_NAME = 'R_50_FPN_UAV_extratest'
RESUME = True
TEST_ONLY = False
START_CHECKPOINT_PATH = None
EXTRA_TEST = True
# START_CHECKPOINT_PATH = './output/R_50_FPN_UAV_last.pth'

# COCO_DATASET_PATH = './datasets/coco_train2017'
# COCO_JSON = os.path.join(COCO_DATASET_PATH, 'annotations/instances_train2017.json')
# COCO_IMGFOLDER = os.path.join(COCO_DATASET_PATH, 'train2017')

DATASET_PATH = '../datasets/UAV-benchmark-M'

VISDRONE_TRAIN_DATASET_PATH = '../datasets/VisDrone2019-DET-train'
VISDRONE_TEST_DATASET_PATH = '../datasets/VisDrone2019-DET-val'

AGH_APTIV_TEST_DATASET_PATH = '../datasets/agh_aptiv'
AGH_APTIV_ANNOTATIONS = '../datasets/agh_aptiv/annotations.json'

COCO_TRAIN_DATASET_PATH = '../datasets/coco2017/images'
COCO_TRAIN_ANNOTATIONS = '../datasets/coco2017/instances_train2017.json'

RESULTS_FILE = 'results.txt'
OUTPUT_DIR = os.path.join('./output', CHECKPOINT_NAME)
NUM_CLASSES = 4
K_FOLDS = 1000    # ignored if kfold data available in checkpoint SPLITS ONLY UAV DATASET
CLASSES_NAMES = ['car', 'truck', 'bus', 'person']
CHECKPOINT_PERIOD = 50
TEST_CONFIDENCE_THRESHOLD = 0.3


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def show_dataset(dataset_dicts):

    for img_dict in dataset_dicts:

        img = cv2.imread(img_dict['file_name'])
        bboxes = [ann['bbox'] for ann in img_dict['annotations']]
        classes = [ann['category_id'] for ann in img_dict['annotations']]
        ids = [ann['target_id'] for ann in img_dict['annotations']]
        gt_img = draw_bboxes(img, bboxes, classes=classes, ids=ids)
        cv2.imshow('test', gt_img)
        print('\nFilename:', img_dict['file_name'], 'img id:', img_dict['image_id']) 
        print('Annotations:')
        for el in img_dict['annotations']:
            print(el)
        if cv2.waitKey(0) == ord('q'):
            break
    cv2.destroyAllWindows()


def setup_cfg():

    cfg = get_cfg()
    cfg.merge_from_file(PROJECT_ROOT + '/configs/mask_rcnn_R_50_FPN_3x.yaml')
    # cfg.MODEL.WEIGHTS = "./pretrained/mask_rcnn_R_50_FPN_3x_original/model_700.pth"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = TEST_CONFIDENCE_THRESHOLD
    cfg.MODEL.MASK_ON = False
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

            if (visualize == 'random' and idx_to_visualize == idx) or visualize == 'all' :
                video_visualizer = VideoVisualizer({'thing_classes': CLASSES_NAMES})
                imgpath = inputs[0]['file_name']
                frame = cv2.imread(imgpath)
                visualized_result = visualize_detections(frame, outputs[0]['instances'], video_visualizer)
                gt_bboxes = [ann['bbox'] for ann in inputs[0]['annotations']]
                visualized_result = draw_bboxes(visualized_result, gt_bboxes)
                
                if visualize == 'all':
                    cv2.imwrite(os.path.join(OUTPUT_DIR, 'test_' + os.path.basename(imgpath)), visualized_result)
                else:
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

            coco_evaluator = COCOeval(cocoGt=coco_gt, cocoDt=coco_dt, iouType='bbox')
            coco_evaluator.evaluate()
            coco_evaluator.accumulate()
            coco_evaluator.summarize()

            return coco_evaluator.stats
            

cfg = setup_cfg()
MAX_ITER = cfg.SOLVER.MAX_ITER
model = build_model(cfg)
# optimizer = build_optimizer(cfg, model)
optimizer = build_finetune_optimizer(cfg, model, to_train=['proposal_generator', 'roi_heads'])
scheduler = build_lr_scheduler(cfg, optimizer)
# uav_dataset_dicts = generate_uav_dataset_dictionaries(DATASET_PATH)
uav_dataset_dicts = []
visdrone_train_dataset_dicts = generate_visdrone_dataset_dictionaries(VISDRONE_TRAIN_DATASET_PATH)
coco_train_dataset_dicts = generate_coco_dataset_dictionaries(COCO_TRAIN_ANNOTATIONS, COCO_TRAIN_DATASET_PATH, allowed_classes=CLASSES_NAMES, category_mapping=COCO_CATEGORY_IDS_TO_UAV)
visdrone_test_dataset_dicts = generate_visdrone_dataset_dictionaries(VISDRONE_TEST_DATASET_PATH)
agh_aptiv_test_dataset_dicts = generate_coco_dataset_dictionaries(AGH_APTIV_ANNOTATIONS, AGH_APTIV_TEST_DATASET_PATH)
# coco_dataset_dicts = generate_coco_dataset_dictionaries(COCO_JSON, COCO_IMGFOLDER, allowed_classes=CLASSES_NAMES, category_mapping=COCO_CATEGORY_IDS_TO_UAV)
show_dataset(coco_train_dataset_dicts)
# show_dataset(uav_dataset_dicts)
# show_dataset(visdrone_train_dataset_dicts)
# show_dataset(agh_aptiv_test_dataset_dicts)

if TEST_ONLY:
    test_checkpoint = os.path.join(OUTPUT_DIR, CHECKPOINT_NAME+'_bestAR.pth')
    chkpt = torch.load(test_checkpoint)
    model.load_state_dict(chkpt['model'])
elif RESUME:
    last_checkpoint = os.path.join(OUTPUT_DIR, CHECKPOINT_NAME+'_last.pth')
    chkpt = torch.load(last_checkpoint)
    print('Resume training from checkpoint ' + last_checkpoint)
    for k, v in chkpt.items():
        print(k, ":", type(v))
    start_iter = chkpt['iteration'] + 1
    # kfold_split = chkpt['kfold_split']
    best_precision = chkpt['best_precision']
    best_recall = chkpt['best_recall']
    # K_FOLDS = chkpt['k_folds']

    model.load_state_dict(chkpt['model'])
    optimizer.load_state_dict(chkpt['optimizer'])
    scheduler.load_state_dict(chkpt['scheduler'])

    with open(os.path.join(OUTPUT_DIR, RESULTS_FILE), 'w') as file:
        file.write(chkpt['training_results'])  
else:
    if START_CHECKPOINT_PATH == None:
        START_CHECKPOINT_PATH = cfg.MODEL.WEIGHTS
    checkpointer = DetectionCheckpointer(model)
    chkpt = checkpointer.load(START_CHECKPOINT_PATH)
    print('Start training with checkpoint ' + START_CHECKPOINT_PATH)
    start_iter = 0
    # kfold = KFold(n_splits=K_FOLDS, shuffle=True)
    # kfold_split = list(kfold.split(uav_dataset_dicts))
    # kfold_split = kfold_split[0]    #CROSS VALIDATION not implemented yet, take first fold
    best_precision = 0
    best_recall = 0
    with open(os.path.join(OUTPUT_DIR, RESULTS_FILE), 'w') as file:
        file.write('\t\tAP\tAP_05\tAP0.75\tAP_s\tAP_m\tAP_l\tAR_1\tAR_10\tAR_100\tAR_s\tAR_m\tAR_l\ttrain_loss\n')  


# print('UAV dataset dicts:', len(uav_dataset_dicts))
# (uav_train_ids, uav_test_ids) = kfold_split
# train_uav_dataset_dicts = [uav_dataset_dicts[i] for i in uav_train_ids]
# test_uav_dataset_dicts = [uav_dataset_dicts[i] for i in uav_test_ids]

print('Loaded {} images for training and {} images for testing'.format(len(visdrone_train_dataset_dicts), len(visdrone_test_dataset_dicts)))


if TEST_ONLY:
    test_dataloader = build_detection_test_loader(cfg, agh_aptiv_test_dataset_dicts)
    do_test(model, test_dataloader, visualize='all')
else:
    if EXTRA_TEST:
        extra_test_dataloader = build_detection_test_loader(cfg, agh_aptiv_test_dataset_dicts)
    test_dataloader = build_detection_test_loader(cfg, visdrone_test_dataset_dicts)
    train_dataloader = build_detection_train_loader(cfg, visdrone_train_dataset_dicts+coco_train_dataset_dicts)
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

            print('Iter. {}/{}:\tloss: {}'.format(iteration, MAX_ITER, losses), end='\r')
            if iteration != 0 and iteration % CHECKPOINT_PERIOD == 0:

                if EXTRA_TEST:
                    extra_test_results = do_test(model, extra_test_dataloader, visualize='all')
                test_results = do_test(model, test_dataloader, visualize='random')
                if test_results is not None:
                    results_string = ['{:.3f}'.format(el) for el in test_results]
                    results_string.append('{:.3f}'.format(losses))
                    if EXTRA_TEST:
                        results_string.append('{:.3f}'.format(extra_test_results[0]))
                        results_string.append('{:.3f}'.format(extra_test_results[8]))
                    results_string = '{}/{}:\t'.format(iteration, MAX_ITER) + '\t'.join(results_string)
                    AP_all = test_results[0]
                    AR_all = test_results[8]
                    if AP_all > best_precision:
                        best_precision = AP_all
                        # results_string += ' Best precision!'
                    if AR_all > best_recall:
                        best_recall = AR_all
                        # results_string += ' Best recall!'
                    with open(os.path.join(OUTPUT_DIR, RESULTS_FILE), 'a') as file:
                        file.write(results_string + '\n')
                    plot_training_results(os.path.join(OUTPUT_DIR, RESULTS_FILE), OUTPUT_DIR, extra_plot=EXTRA_TEST)
                    

                with open(os.path.join(OUTPUT_DIR, RESULTS_FILE), 'r') as file:
                    checkpoint_dict = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': iteration,
                        # 'kfold_split': kfold_split,
                        # 'k_folds': K_FOLDS,
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