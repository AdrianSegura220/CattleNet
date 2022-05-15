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

# model

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# convnext_tiny = models.convnext_tiny(pretrained=True)
# convnext_tiny.classifier[2] = nn.Sequential(
#     nn.Linear(768,4096,bias=True),
#     nn.Sigmoid()
# )   

# convnext_tiny.to(device) # load NN to cuda or cpu

"""
    TO TRY NEXT:
     - Try weight decay in optimizer: weight_decay=0.0005
"""
                                 

#hyperparams
in_channel = 3
batch_size = 64
num_epochs = 5

# instantiate SNN
model = CattleNet().cpu()

# loss function
criterion = ContrastiveLoss()

# setup optimizer (use Adam technique to optimize parameters (GD with momentum and RMS prop))
optimizer = optim.Adam(model.parameters()) # by default: learning rate = 0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0

dataset = CustomImageDataset(dataset_folder='../../dataset/Raw/RGB (320 x 240)/',img_dir='../../dataset/Raw/Combined/',transform=transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]))
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

def train():
    loss = []
    counter = []
    iteration_number = 0
    for epoch in range(1,num_epochs):
        for imgs, labels in data_loader:
            #### do something with images ...
            optimizer.zero_grad()
            # out1,out2 = model(img1,img2)
            # loss_contrastive = criterion(out1,out2,label)
            # loss_contrastive.backwards()
            optimizer.step()
        # print("Epoch {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
        iteration_number += 10
        counter.append(iteration_number)
        # loss.append(loss_contrastive.item())
    plt.plot(counter,loss)
    plt.show()
    return model

device = torch.device('cuda' if th.cuda.is_available() else 'cpu')
model = train()
torch.save(model.state_dict(), "model.pt")
print("Model Saved Successfully") 