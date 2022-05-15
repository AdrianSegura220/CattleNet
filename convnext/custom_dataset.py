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

"""
    Custom dataset to use with CattleNet.
    dataset_folder: path to dataset containing folders having label and images of such labeled cow ('../../dataset/Raw/RGB (320 x 240)/')
    img_dir: directory containing all images combined
    transform: a transformation to apply to each image when extracting using __getitem__()
    target_transform: a transformation to apply to label of each image when extracting using __getitem__()
"""
class CustomImageDataset(Dataset):
    def __init__(self, dataset_folder, img_dir, transform=None, target_transform=None) -> None:
        # super().__init__() no superconstructor
        generate_annotations(dataset_folder)
        self.img_labels = pd.read_csv('annotations.csv')
        self.img_dir = img_dir
        self.transform = transform
        self.counts = {}
        self.countPerSample()
        self.target_transform = target_transform

    """
        Have count of how many images exist per label
    """
    def countPerSample(self):
        curr = 0
        prev = 0
        for i in range(0,self.__len__()):
            if self.img_labels.iloc[i,2] in self.counts:
                self.counts[self.img_labels.iloc[i,2]] += 1
            else:
                self.counts[self.img_labels.iloc[i,2]] = 0

    def __len__(self):
        return len(self.img_labels)

    """
        Return image as float tensor
    """
    def __getitem__(self, idx):
        # path to image being picked
        im1name = ''
        im2name = ''
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])
        same_class = random.randint(0,1) 
        
        if same_class:
            for i in range(0,self.__len__()):
                if self.img_labels.iloc[i,2] == self.img_labels.iloc[idx, 2]:
                    if self.counts[self.img_labels.iloc[idx, 2]] > 1:
                        selectedImage = random.randint(0,(self.counts[self.img_labels.iloc[idx, 2]]-1)) # select one of the pictures randomly
                        im2name = self.img_labels.iloc[i+selectedImage, 1]
                    else:
                        selectedImage = 0
                    image2 = read_image(os.path.join(self.img_dir, self.img_labels.iloc[i+selectedImage, 1])).float() # selected image should be of same cow
                    label2 = self.img_labels.iloc[i+selectedImage, 2]
                    im2name = self.img_labels.iloc[i+selectedImage, 1]
                    break
        else:
            rand_idx = random.randint(0,self.__len__()-1)
            image2 = read_image(os.path.join(self.img_dir, self.img_labels.iloc[rand_idx, 1])).float() # choose a random image
            label2 = self.img_labels.iloc[rand_idx, 2]
        # read the RGB image (i.e. load it to a 3x240x320 tensor)
        image = read_image(img_path).float()

        # read label of the image (i.e. which cow it is)
        label = self.img_labels.iloc[idx, 2]

        # if transformation was given when instantiating dataset, apply it (same for label transform (target_transform))
        if self.transform:
            image = self.transform(image)
            image2 = self.transform(image2)
        if self.target_transform:
            label = self.target_transform(label)

        print('label1: ',label)
        print('im1 path: ', img_path)
        print('label2: ',label2)
        print('im1 path: ', im2name)
        return image,image2,label,label2

