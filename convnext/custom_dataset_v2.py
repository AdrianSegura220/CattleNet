import os
import pandas as pd
from soupsieve import select
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
import random
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from preprocess import generate_annotations
import sys
sys.path.insert(0, '/home/adriansegura/Desktop/RUG/CattleNet/data_manipulation')

"""
    Custom dataset to use with CattleNet.
    dataset_folder: path to dataset containing folders having label and images of such labeled cow ('../../dataset/Raw/RGB (320 x 240)/')
    img_dir: directory containing all images combined
    transform: a transformation to apply to each image when extracting using __getitem__()
    target_transform: a transformation to apply to label of each image when extracting using __getitem__()
"""
class CustomImageDatasetV2(Dataset):
    def __init__(self, dataset_folder, img_dir, transform=None, target_transform=None,testing=False) -> None:
        # super().__init__() no superconstructor
        # generate_annotations(dataset_folder)
        # self.train_size = train_size
        if testing:
            self.img_labels = pd.read_csv('annotations_testing.csv')
        else:
            self.img_labels = pd.read_csv('annotations_training.csv')
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    """
        Return image as float tensor
    """
    def __getitem__(self, idx):
        # path to image being picked
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])
        image2 = (read_image(os.path.join(self.img_dir, self.img_labels.iloc[idx, 3])).float())/255.0
        label2 = self.img_labels.iloc[idx, 4]
        image = (read_image(img_path).float())/255.0
        label = self.img_labels.iloc[idx, 2]

        # print('label 1: ',label)
        # print('path 1: ',img_path)
        # print('label 2: ',label2)
        # print('path 2:', os.path.join(self.img_dir, self.img_labels.iloc[idx, 3]))
        # if transformation was given when instantiating dataset, apply it (same for label transform (target_transform))
        if self.transform:
            # print("transforming")
            image = self.transform(image)
            image2 = self.transform(image2)
            # print('IMG1 size: ', image.size())
            # print('IMG2 size: ', image2.size())
        if self.target_transform:
            label = self.target_transform(label)

        return image,image2,label,label2

