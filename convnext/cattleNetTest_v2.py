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
import custom_dataset
from torch.utils.data import DataLoader

class CattleNetV2(nn.Module):
    def __init__(self,freezeLayers=False) -> None:
        super(CattleNetV2,self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3,32,(15,15)),
            nn.ReLU(),
            nn.MaxPool2d(2,2) # 112 x 112 x 32
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32,64,(10,10)),
            nn.ReLU(),
            nn.MaxPool2d(2,2) # 51 x 51 x 64
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64,128,(7,7)),
            nn.ReLU(),
            nn.MaxPool2d(2,2) # 22 x 22 x 128
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128,128,(4,4)),
            nn.ReLU(),
            nn.MaxPool2d(2,2) # 9 x 9 x 128
        )
        self.block5 = nn.Sequential( #check all net
            nn.Conv2d(128,256,(4,4)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(12544,4096,bias=True),
            nn.Sigmoid()
        )

        # self.convnext_tiny = models.convnext_tiny(pretrained=True)
        # if freezeLayers:
        #     self.freeze_layers()
        # self.convnext_tiny.classifier[2] = nn.Linear(768,4096,bias=True)
        # self.convnext_tiny = nn.Sequential(self.convnext_tiny,nn.Sigmoid())
        

    def forward_once(self,input):
        # print('input:',input.size())
        b1 = self.block1(input)
        # print('b1: ',b1)
        b2 = self.block2(b1)
        # print('b2: ',b2)
        b3 = self.block3(b2)
        # print('b3: ',b3)
        b4 = self.block4(b3)
        # print('b4: ',b4)
        # exit()
        feature_vect = self.block5(b4)
        # x = x = x.flatten(start_dim=1)
        return feature_vect
        # return self.convnext_tiny.features(input)

    def forward(self,input1,input2):
        out1 = self.forward_once(input1)
        out2 = self.forward_once(input2)

        return out1,out2