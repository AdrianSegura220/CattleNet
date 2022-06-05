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

class CattleNet(nn.Module):
    def __init__(self,freezeLayers=False,cross_entropy = False,embedding_size = 4096) -> None:
        super(CattleNet,self).__init__()
        self.convnext_tiny = models.convnext_tiny(pretrained=True)

        if freezeLayers:
            self.freeze_layers()

        # ORIGINALLY: 
        # self.convnext_tiny.classifier[2] = nn.Linear(768,4096,bias=True)

        self.convnext_tiny = nn.Sequential(
            self.convnext_tiny,
            nn.Linear(1000,2048,bias=True),
            nn.Sigmoid()
        )
        self.cross_entropy = cross_entropy

        # only used when cross_entropy = True
        # self.diff_layer = nn.Sequential( # difference layer
        #     nn.Linear(embedding_size,1),
        #     nn.Sigmoid()
        # )



    def unfreeze_layers(self):
        for param in self.convnext_tiny.parameters():
            param.requires_grad = True
    """
        Used to freeze pre-trained layers if indicated.
        Otherwise default is false (i.e. they are also
        tuned during training)
    """
    def freeze_layers(self):
        for param in self.convnext_tiny.parameters():
            param.requires_grad = False

        # keep last native layer trainable (not original setup)
        for param in self.convnext_tiny.classifier[2].parameters():
            param.requires_grad = True

    def forward_once(self,input):
        x = self.convnext_tiny(input)
        # x = x = x.flatten(start_dim=1)
        return x
        # return self.convnext_tiny.features(input)

    def forward(self,input1,input2):
        out1 = self.forward_once(input1)
        out2 = self.forward_once(input2)


        if self.cross_entropy:
            res = self.diff_layer(torch.abs(torch.sub(out1,out2)))
            # print('RES SIZE: ',res.size())s
            return res
        else:
            return out1,out2