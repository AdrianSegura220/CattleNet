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

# model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



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
batch_size = 8
num_epochs = 60

# instantiate SNN
model = CattleNet(freezeLayers=True).to(device)
# print(model)
# exit()

# loss function
criterion = ContrastiveLoss()

# setup optimizer (use Adam technique to optimize parameters (GD with momentum and RMS prop))
# optimizer = optim.Adam(model.parameters()) # by default: learning rate = 0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0
params = list(model.convnext_tiny.classifier.parameters())+list(model.classifier_layer.parameters())
optimizer = optim.Adam(params) 
dataset = CustomImageDataset(dataset_folder='../../dataset/Raw/RGB (320 x 240)/',img_dir='../../dataset/Raw/Combined/',transform=transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]))

# setup training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
len(dataset)

data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def train():
    loss = []
    counter = []
    iteration_number = 0
    epoch_loss = 0.0
    iterations_loop = 0
    for epoch in range(1,num_epochs):
        loop = tqdm(data_loader,leave=False,total=len(data_loader))
        epoch_loss = 0.0
        iterations_loop = 0
        for data in loop:
            label = 0
            #### do something with images ...
            optimizer.zero_grad()
            imgs1 = data[0].to(device)
            imgs2 = data[1].to(device)
            labels1 = data[2].to(device)
            labels2 = data[3].to(device)
            out1,out2 = model(imgs1,imgs2)
            # print('d2: ',data[2],'; d3: ',data[3])
            label = (labels1 != labels2).float()
            loss_contrastive = criterion(out1,out2,label)
            loss_contrastive.backward()
            optimizer.step()
            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(loss=loss_contrastive.item())
            epoch_loss += loss_contrastive.item()
            iterations_loop += 1
        epoch_loss /= iterations_loop
        print("Epoch {}\n Current loss {}\n".format(epoch,epoch_loss))
        iteration_number += 10
        counter.append(iteration_number)
        loss.append(epoch_loss)
        # loss.append(loss_contrastive.item())
    plt.plot(counter,loss)
    plt.show()
    return model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.train()
print("Starting training")
model = train()
torch.save(model.state_dict(), "model2_003lr.pt")
print("Model Saved Successfully") 