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
class TriplesDataset(Dataset):
    def __init__(self, dataset_folder, img_dir, transform=None, target_transform=None) -> None:
        # super().__init__() no superconstructor
        generate_annotations(dataset_folder)
        # self.train_size = train_size
        self.img_labels = pd.read_csv('annotations.csv')
        self.img_dir = img_dir
        self.transform = transform
        self.counts = {}
        self.countPerSample()
        self.target_transform = target_transform
        self.embeddings = {}
        self.is_update_embeddings = False # false by default

    """
        Have count of how many images exist per label
    """
    def countPerSample(self):
        for i in range(0,self.__len__()):
            if self.img_labels.iloc[i,2] in self.counts:
                self.counts[self.img_labels.iloc[i,2]] += 1
            else:
                self.counts[self.img_labels.iloc[i,2]] = 0

    def __len__(self):
        return len(self.img_labels)

    """
        1. select a random positive example of such label
        2. select a random example that violates constraint of triplet loss (i.e. a hard example) !! need to find embeddings of images
        3. return triple
    """
    def __getitem__(self, idx):
        # CONTINUE DEFINING BOTH IF AND ELSE PARTS
        if self.is_update_embeddings:
            pass 
            return 
        else:
            # path to image being picked
            im1name = ''
            im2name = ''
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])
            # read the RGB image (i.e. load it to a 3x240x320 tensor)
            anchor = (read_image(img_path).float())/255.0
            # read label of the image (i.e. which cow it is)
            label = self.img_labels.iloc[idx, 2]    

            # select positive example
            for i in range(0,self.__len__()):
                if self.img_labels.iloc[i,2] == label:
                    if self.counts[label] > 1: # if existing images of such label are more than 1
                        selectedImage = random.randint(0,(self.counts[label]-1)) # select one of the pictures randomly
                    else: # else if there is only one image from that example
                        selectedImage = 0
                    positive_example = (read_image(os.path.join(self.img_dir, self.img_labels.iloc[i+selectedImage, 1])).float())/255.0 # selected image should be of same cow
                    # anchor_label = self.img_labels.iloc[i+selectedImage, 2]
                    break

            # select negative example
            is_condition_met = False
            while not is_condition_met:
                rand_idx = random.randint(0,self.__len__()-1)
                negative_label = self.img_labels.iloc[rand_idx, 2]
                if negative_label != label:
                    negative_example = (read_image(os.path.join(self.img_dir, self.img_labels.iloc[rand_idx, 1])).float())/255.0 # choose a random image
                
            

            # if transformation was given when instantiating dataset, apply it (same for label transform (target_transform))
            if self.transform:
                # print("transforming")
                anchor = self.transform(anchor)
                positive_example = self.transform(positive_example)
                negative_example = self.transform(negative_example)
                # print('IMG1 size: ', image.size())
                # print('IMG2 size: ', image2.size())

            
            return anchor,image2,label,label2

