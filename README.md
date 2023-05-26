# GreenPose

## Requirements

* Ubuntu 16.04
* Python 3.6
* PyTorch 0.4.1 (should also work with 1.0, but not tested)

## Prerequisites

1. Download COCO 2017 dataset: [http://cocodataset.org/#download](http://cocodataset.org/#download) (train, val, annotations) and unpack it to `<COCO_HOME>` folder.
2. Install requirements `pip install -r requirements.txt`

## Training

Training consists of 3 steps (given AP values for full validation dataset):
* Training from MobileNet weights. Expected AP after this step is ~38%.
* Training from weights, obtained from previous step. Expected AP after this step is ~39%.
* Training from weights, obtained from previous step and increased number of refinement stages to 3 in network. Expected AP after this step is ~40% (for the network with 1 refinement stage, two next are discarded).

1. Download pre-trained MobileNet v1 weights `mobilenet_sgd_68.848.pth.tar` from: [https://github.com/marvis/pytorch-mobilenet](https://github.com/marvis/pytorch-mobilenet) (sgd option). If this doesn't work, download from [GoogleDrive](https://drive.google.com/file/d/18Ya27IAhILvBHqV_tDp0QjDFvsNNy-hv/view?usp=sharing).

2. Convert train annotations in internal format. Run `python scripts/prepare_train_labels.py --labels <COCO_HOME>/annotations/person_keypoints_train2017.json`. It will produce `prepared_train_annotation.pkl` with converted in internal format annotations.

   [OPTIONAL] For fast validation it is recommended to make *subset* of validation dataset. Run `python scripts/make_val_subset.py --labels <COCO_HOME>/annotations/person_keypoints_val2017.json`. It will produce `val_subset.json` with annotations just for 250 random images (out of 5000).

3. To train from MobileNet weights, run `python train.py --train-images-folder <COCO_HOME>/train2017/ --prepared-train-labels prepared_train_annotation.pkl --val-labels val_subset.json --val-images-folder <COCO_HOME>/val2017/ --checkpoint-path <path_to>/mobilenet_sgd_68.848.pth.tar --from-mobilenet`

4. Next, to train from checkpoint from previous step, run `python train.py --train-images-folder <COCO_HOME>/train2017/ --prepared-train-labels prepared_train_annotation.pkl --val-labels val_subset.json --val-images-folder <COCO_HOME>/val2017/ --checkpoint-path <path_to>/checkpoint_iter_420000.pth --weights-only`

5. Finally, to train from checkpoint from previous step and 3 refinement stages in network, run `python train.py --train-images-folder <COCO_HOME>/train2017/ --prepared-train-labels prepared_train_annotation.pkl --val-labels val_subset.json --val-images-folder <COCO_HOME>/val2017/ --checkpoint-path <path_to>/checkpoint_iter_280000.pth --weights-only --num-refinement-stages 3`. We took checkpoint after 370000 iterations as the final one.
