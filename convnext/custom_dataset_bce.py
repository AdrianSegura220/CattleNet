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
from preprocess import generate_annotations,generate_annotations_direct

"""
    Custom dataset to use with CattleNet.
    dataset_folder: path to dataset containing folders having label and images of such labeled cow ('../../dataset/Raw/RGB (320 x 240)/')
    img_dir: directory containing all images combined
    transform: a transformation to apply to each image when extracting using __getitem__()
    target_transform: a transformation to apply to label of each image when extracting using __getitem__()
"""
class CustomImageDatasetBCE(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None) -> None:
        # super().__init__() no superconstructor
        generate_annotations_direct(img_dir,'training_annotations')
        # self.train_size = train_size
        self.img_labels = pd.read_csv('training_annotations.csv')
        self.img_dir = img_dir
        self.transform = transform
        self.counts = {}
        self.countPerSample()
        self.target_transform = target_transform

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
        return len(self.img_labels)-3

    """
        Return image as float tensor
    """
    def __getitem__(self, idx):
        # path to image being picked
        im1name = ''
        im2name = ''
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])
        # same_class = random.randint(0,1) 
        same_class = random.choice([0,1]) # used to approximately select around 50% of samples to be equal and 50% to be different
        
        if same_class:
            for i in range(0,self.__len__()):
                if self.img_labels.iloc[i,2] == self.img_labels.iloc[idx, 2]:
                    if self.counts[self.img_labels.iloc[idx, 2]] > 1:
                        selectedImage = random.randint(0,(self.counts[self.img_labels.iloc[idx, 2]]-1)) # select one of the pictures randomly
                    else:
                        selectedImage = 0
                    image2 = (read_image(os.path.join(self.img_dir, self.img_labels.iloc[i+selectedImage, 1])).float())/255.0 # selected image should be of same cow
                    label2 = self.img_labels.iloc[i+selectedImage, 2]
                    break
        else:
            # total_to_use = self.__len__() if self.train_size == -1 else self.train_size # set max num used for training purposes
            rand_idx = random.randint(0,self.__len__()-1)
            image2 = (read_image(os.path.join(self.img_dir, self.img_labels.iloc[rand_idx, 1])).float())/255.0 # choose a random image
            label2 = self.img_labels.iloc[rand_idx, 2]
        # read the RGB image (i.e. load it to a 3x240x320 tensor)
        image = (read_image(img_path).float())/255.0

        # read label of the image (i.e. which cow it is)
        label = self.img_labels.iloc[idx, 2]

        # if transformation was given when instantiating dataset, apply it (same for label transform (target_transform))
        if self.transform:
            # print("transforming")
            image = self.transform(image)
            image2 = self.transform(image2)
        if self.target_transform:
            label = self.target_transform(label)

        
        return image,image2, 1.0 if label == label2 else 0.0 




# generate csv for training and for validation
# complete selection of negative elements
# send email

class CustomImageDataset_Validation(Dataset):
    def __init__(self, img_dir, n, transform=None, target_transform=None) -> None:
        # super().__init__() no superconstructor
        generate_annotations_direct(img_dir,'validation_annotations')
        # self.train_size = train_size
        self.img_labels = pd.read_csv('validation_annotations.csv')
        self.img_dir = img_dir
        self.transform = transform
        self.n_size = n if n >= 2 else 2 # minimum is 2 (one positive for anchor and a negative)
        self.counts = {}
        self.countPerSample()
        self.target_transform = target_transform

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
        return len(self.img_labels)-1

    """
        Get one image and n for comparison
    """
    def __getitem__(self, idx):        
        anchor = (read_image(os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])).float())/255.0
        anchor_label = self.img_labels.iloc[idx, 2]
        
        # select second image from same class
        if self.counts[self.img_labels.iloc[idx, 2]] > 1:
            selectedImage = random.randint(0,(self.counts[self.img_labels.iloc[idx, 2]]-1)) # select one of the pictures randomly
        else:
            selectedImage = 0
        
        print('idx: ', idx)
        print('selected image: ', selected_image+idx)

        # create tensor of (n, 3, dimension 1 of imgs, dimension 2 of imgs) dimensions to store all images of certain size (e.g. (8,3,240,240)) .
        images = torch.Tensor(self.n_size,3,anchor.size()[1],anchor.size()[2])
        final_images = torch.Tensor(self.n_size,3,anchor.size()[1],anchor.size()[1])
        # create tensor for labels (n,1) i.e. a column vector
        labels = torch.zeros(self.n_size,1)

        # put positive example in first slot and set label to positive one for same slot
        images[0] = (read_image(os.path.join(self.img_dir, self.img_labels.iloc[idx+selectedImage, 1])).float())/255.0 # selected image should be of same cow
        labels[0] = 1.0

        # selected starts from 1 as 1 image (the positive example) has been selected already
        selected = 1

        # select n-1 negative example images for verification purposes
        while selected < self.n_size:
            random_idx = random.randint(0,self.__len__()-1) # select a random index
            
            # check if the chosen image's index belongs to a different cow
            if anchor_label != self.img_labels.iloc[random_idx, 2]:
                images[selected] = (read_image(os.path.join(self.img_dir,self.img_labels.iloc[random_idx, 1])).float())/255.0
                selected += 1
        
        # generate list of shuffled indices to shuffle images and labels in a pair-wise fashion
        shuffled_indices = torch.randperm(self.n_size).to(torch.int64)
	print(shuffled_indices)        
        # permute elements from imgs and labels in same order
        for i in range(0,self.n_size):
            temp_img = images[i]
            temp_label = labels[i]
            images[i] = images[shuffled_indices[i]]
            images[shuffled_indices[i]] = temp_img
            labels[i] = labels[shuffled_indices[i]]
            labels[shuffled_indices[i]] = temp_label

        figures = []
        # if transformation was given when instantiating dataset, apply it
        if self.transform:
            anchor = self.transform(anchor)
            for i in range(0,self.n_size):
                transform = self.transform(images[i])
                final_images[i] = transform
                # figures.append(plt.figure(figsize=(10, 7)))
                # figures[i].add_subplot(2, 2, 1)
                # plt.imshow(anchor.permute(1,2,0))
                # plt.axis('off')
                # plt.title('anchor')
                # figures[i].add_subplot(2, 2, 2)
                # plt.imshow(transform.permute(1,2,0))
                # plt.axis('off')
                # plt.title('positive' if labels[i] == 1 else 'negative')

        # plt.show()


        return anchor,final_images,labels
