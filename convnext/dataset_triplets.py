import os
import pandas as pd
from sklearn.utils import shuffle
from soupsieve import select
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
import random

class SimpleTriplet(Dataset):
    def __init__(self, img_dir, transform=None,annotations_csv = ''):
        self.img_labels = pd.read_csv(annotations_csv)
        self.img_dir = img_dir
        self.transform = transform
        self.counts = {}
        self.countPerSample()

    """
        Have count of how many images exist per label
    """
    def countPerSample(self):
        for i in range(0,self.__len__()):
            if self.img_labels.iloc[i,2] in self.counts:
                self.counts[self.img_labels.iloc[i,2]][0] += 1
            else:
                self.counts[self.img_labels.iloc[i,2]] = [0,i]

    def __len__(self):
        return len(self.img_labels)-1

    """
        Basically select self.batch_size number of images in total. Per image, 
    """
    def __getitem__(self, idx):
        count_selected = 0
        image_at_idx_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])
        image_at_idx = (read_image(image_at_idx_path).float())/255.0 #used as dummy to extract tensor dimensions
        label_at_idx = self.img_labels.iloc[idx, 2]

        return self.transform(image_at_idx) if self.transform else image_at_idx,label_at_idx