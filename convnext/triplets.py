import os
import pandas as pd
from sklearn.utils import shuffle
from soupsieve import select
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
import random

class TripletSet(Dataset):
    def __init__(self, img_dir, transform=None,annotations_csv = '',model=None,batch_size=64):
        self.img_labels = pd.read_csv(annotations_csv)
        self.img_dir = img_dir
        self.transform = transform
        self.counts = {}
        self.model = model # used to compute embeddings everytime
        self.batch_size = batch_size
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

        embeddings = torch.Tensor(self.batch_size,3,image_at_idx.size()[1],image_at_idx.size()[2])
        # create tensor for labels (n,1) i.e. a column vector
        embedding_labels = torch.zeros(self.batch_size,1)

        i = 0
        selected_idx = 0
        while i < self.__len__() and count_selected < self.batch_size:
            if random.randint(0,1) == 1: # then select image from same label
                for j in range(0,self.__len__()):
                    if self.img_labels.iloc[j, 1] != self.img_labels.iloc[i, 1] and self.img_labels.iloc[j, 2] == self.img_labels.iloc[i, 2]:
                        selected_idx = j
            else:
                selected_idx = random.randint(0,self.__len__()-1)
                while selected_idx == self.img_labels.iloc[i, 2]: # randomly sample index until the selected image is different
                    selected_idx = random.randint(0,self.__len__()-1)


            image_at_idx_path = os.path.join(self.img_dir, self.img_labels.iloc[selected_idx, 1])
            embeddings[count_selected] = (read_image(image_at_idx_path).float())/255.0
            embedding_labels[count_selected] = self.img_labels.iloc[selected_idx, 2]
            count_selected += 1
            i += 1


        # computation of embeddings for mini-batch
        output = torch.Tensor()
        self.model.eval()
        with torch.no_grad():
            output, dummytensor = self.model(embeddings,embeddings)

        # compute pairwise distance matrix to define triplets
        dot_embeddings = torch.matmul(output,torch.transpose(output,0,1))

        diagonal = torch.diagonal(dot_embeddings) # obtain diagonal (contains the squared d)

        # if transformation was given when instantiating dataset, apply it (same for label transform (target_transform))
        if self.transform:
            # print("transforming")
            image = self.transform(image)

        return image,label