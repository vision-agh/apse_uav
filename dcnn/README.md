This is the instruction for running our fine-tuned network used for detecting vehicles from above as in [2022 JSPS paper](https://link.springer.com/article/10.1007/s11265-021-01734-3).

It is based on https://github.com/facebookresearch/detectron2 and https://github.com/NegatioN/OnlineMiningTripletLoss.

Tested on:
- Ubuntu 18.04, Ubuntu 20.04
- cuda 10.2
- Python 3.8.5
- Pytorch 1.5.0

### Dependencies
Using [virtualenv](https://virtualenv.pypa.io/en/latest/installation.html) is highly recommended

detectron2 dependencies:
```sh
pip install torch torchvision opencv-python
```
preferably outside of this repository:
```sh
git clone https://github.com/facebookresearch/detectron2
cd detectron2
git checkout 4ee7971        #tested on this commit - the library is still in development
cd ..
python -m pip install -e detectron2
```

### Original mask r-cnn demo

- Download pretrained model from detectron2 model zoo: https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md
-- For example, [mask r-cnn R50-FPN](https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl)
- Create ```pretrained``` directory in project root
- Put model in ```rcnn_tracker/pretrained/mask_rcnn_R50_FPN_3x/model_final_f10217.pkl```
- Run in project root: ```python scripts/tests/standard_rcnn_detector_test.py```
- Note. Config file should be compatible with model, check ```def setup_cfg()``` function


### Training (fine-tuning on UAV-benchmark)

- UAV-benchmark project site: https://sites.google.com/site/daviddo0323/projects/uavdt
- Create ```datasets``` directory in project root
- Download images: https://drive.google.com/open?id=1m8KA6oPIRK_Iwt9TYFquC87vBc_8wRVc
- Extract them to ```rcnn_tracker/datasets/UAV-benchmark-M```
- Download toolkit: https://drive.google.com/open?id=19498uJd7T9w4quwnQEy62nibt3uyT9pq
- Put ```GT``` folder from toolkit to dataset directory, so the directory tree looks like this:
```
\rcnn_tracker
    \configs
    \engines
    \pretrained
    ...
    \datasets
        \UAV-benchmark-M
            \GT
                M0101_gt.txt
                M0101_gt_ignore.txt
                ...
            \M0101
            \M0201
            ...
```
- Run ```python scripts/train/finetune_faster_rcnn_aerial.py```

