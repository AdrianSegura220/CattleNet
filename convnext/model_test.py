from __future__ import print_function
from __future__ import division
from turtle import forward
from torchvision import datasets, models, transforms 
from torchvision import datasets, transforms as T
from contrastive_loss import ContrastiveLoss
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time
import os
import copy
from custom_dataset import CustomImageDataset
from torch.utils.data import DataLoader
from cattleNetTest import CattleNet
from tqdm import tqdm

"""
    Go through each image, do forward pass through CNN to generate feature vectors
    and pair them with their labels in a dict.
    For each label, revise whether own images have the smallest euclidean distance
    compared to other images
"""
def test(feature_vector: torch.Tensor):
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#../../BachelorsProject/Trainings/bs8_adam_lrem3_05_17/
model = CattleNet()
model.load_state_dict(torch.load('model_bs8_epoch21_adam_lr1em3_frozen.pt'))
model.to(device)
model.eval()

dataset = CustomImageDataset(dataset_folder='../../dataset/Raw/RGB (320 x 240)/',img_dir='../../dataset/Raw/Combined/',transform=transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]))

# setup training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

for data in data_loader:
    imgs1 = data[0].to(device)
    imgs2 = data[1].to(device)
    labels1 = data[2].to(device)
    labels2 = data[3].to(device)
    out1,out2 = model(imgs1,imgs2)
    print('img1: ',data[2],' img2: ',data[3])
    euclidean = torch.nn.PairwiseDistance(2)
    output = euclidean(out1,out2)
    print('euclidean distance: ',output)
    print('')
