from __future__ import print_function
from __future__ import division
from turtle import forward
from torchvision import datasets, models, transforms
from torchvision import datasets, transforms as T
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

from websockets import Data
import custom_dataset
from torch.utils.data import DataLoader

# batch_size = 32
# img_dir = '../../dataset/Raw/Combined' # directory holding all images


# dataset = custom_dataset.CustomImageDataset(img_dir,transform=transforms.ToTensor())

# train_set, test_set = torch.utils.data.random_split(dataset,[1004,334])

# train_loader = DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)
# test_loader = DataLoader(dataset=test_set,batch_size=batch_size,shuffle=True)

class ContrastiveLoss(nn.Module):
    def __init__(self,margin=1.0) -> None:
        super(ContrastiveLoss,self).__init__()
        self.margin = margin
    
    def forward(self, t1: torch.Tensor,t2: torch.Tensor,label):
        t12 = t2-t1
        euclidean = torch.sqrt(torch.sum(t12.pow(2)))
        margin = 1
        loss = (1-label)*(1/2)*(euclidean)+label*(1/2)*torch.pow(torch.max(0,margin-euclidean),2)
        return loss


# model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparams
in_channel = 3
learning_rate = 0.001
batch_size = 64
num_epochs = 5

convnext_tiny = models.convnext_tiny(pretrained=True)
convnext_tiny.classifier[2] = nn.Sequential(
    nn.Linear(768,4096,bias=True),
    nn.Sigmoid()
)   

print(convnext_tiny)
exit()
normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])



