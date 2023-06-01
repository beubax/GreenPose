import argparse
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.nn import DataParallel
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.coco import CocoTrainDataset
from datasets.transformations import ConvertKeypoints, Scale, Rotate, CropPad, Flip

stride = 1
sigma = 7
path_thickness = 1

dataset = CocoTrainDataset("prepared_train_annotation.pkl", "COCO/train2017",
                               stride, sigma, path_thickness,
                               transform=transforms.Compose([
                                   ConvertKeypoints(),
                                #    Scale(),
                                #    Rotate(pad=(128, 128, 128)),
                                #    CropPad(pad=(128, 128, 128)),
                                #    Flip()
                                ])
)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

batch_data = next(iter(train_loader))

def draw_heatmaps(heatmaps, image, index):
    img = image
    #print(img.max(), img.min(), img.std(), img.mean())
    img = np.array(255*img.transpose(1, 2, 0), dtype = np.uint8)
    img = cv2.resize(img, (heatmaps.shape[1], heatmaps.shape[1]))
    #print(img.shape, img.max(), img.min(), img.mean(), img.std())
    #print(img.shape)W
    #print(heatmaps.shape[0])
    for i in range(heatmaps.shape[0]):
        #current = cv2.applyColorMap(heatmaps[i, :, :], cv2.COLORMAP_JET)
        current = heatmaps[i, :, :]
        current = cv2.resize(current, (img.shape[0], img.shape[1]))
        #print(current.shape)
        #print(current.mean())
        #print(current.std())
        #print(img.max())
        plt.imshow(img)
        plt.imshow(current, alpha = 0.5)
        plt.savefig(str(index) + '_' + str(i) + '.png')
    print("saved", str(index))

print(batch_data['keypoint_maps'][0].shape)

# Convert pytorch tensor to numpy array



print(batch_data['image'][0].shape)

draw_heatmaps(batch_data['keypoint_maps'][0].numpy(), batch_data['image'][0].numpy(), 0)
# draw_heatmaps(dataset[0]['keypoint_maps'], dataset[0]['image'], 0)
# print(dataset[10]['keypoint_mask'].shape)
# print(dataset[10]['image'].shape)
# unique, counts = np.unique(dataset[10]['keypoint_mask'], return_counts=True)

# print(dict(zip(unique, counts)))