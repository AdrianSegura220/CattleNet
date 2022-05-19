from __future__ import print_function
from __future__ import division
from turtle import forward
from torchvision import datasets, models, transforms 
from torchvision import datasets, transforms as T
from contrastive_loss import ContrastiveLoss
from torch.optim.lr_scheduler import StepLR
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time
import datetime
import os
import copy
import wandb
from custom_dataset import CustomImageDataset
from torch.utils.data import DataLoader
from cattleNetTest import CattleNet
from tqdm import tqdm


# wandb setup (logging progress to online platform)
wandb.init(project="cattleNet-arch1", entity="adriansegura220")
#HELLO BORA YILMAZ

# setup device type
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# results folders
path_to_results = '../../BachelorsProject/Trainings/'   

#hyperparams
lrDecay = 1
step_lr = 5
lr=1e-3
in_channel = 3
batch_size = 8
num_epochs = 20


wandb.config = {
  "learning_rate": lr,
  "epochs": num_epochs,
  "batch_size": batch_size
}

# instantiate SNN
model = CattleNet(freezeLayers=True)
model.to(device)
print(model)
# print(model)

# loss function
criterion = ContrastiveLoss()

# setup optimizer (use Adam technique to optimize parameters (GD with momentum and RMS prop))
# optimizer = optim.Adam(model.parameters()) # by default: learning rate = 0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0
params = model.parameters()
optimizer = optim.Adam(params) 
dataset = CustomImageDataset(dataset_folder='../../dataset/Raw/RGB (320 x 240)/',img_dir='../../dataset/Raw/Combined/',transform=transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]))

# setup learning rate scheduler
scheduler = StepLR(optimizer, step_size=step_lr, gamma=0.1)

# setup training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
len(dataset)

data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def train():
    min_loss = float('inf')
    loss = []
    counter = []
    iteration_number = 0
    epoch_loss = 0.0
    iterations_loop = 0
    # create directory for current training results
    final_path = os.path.join(path_to_results,'model_InitialLR{}_lrDecay{}wStep{}_trainSize{}_testSize{}_datetime{}-{}H{}M{}'.format(lr,lrDecay,step_lr,train_size,test_size,datetime.datetime.today().day,datetime.datetime.today().month,datetime.datetime.today().hour,datetime.datetime.today().minute))
    os.mkdir(final_path)

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
        scheduler.step()
        epoch_loss /= iterations_loop
        curr_lr = optimizer.state_dict()['param_groups'][0]['lr']

        #print details of elapsed epoch
        print("lr {}".format(curr_lr))
        print("Epoch {}\n Current loss {}\n".format(epoch,epoch_loss))
        wandb.log({"loss": epoch_loss})

        # maintain epochs in scales of 10
        iteration_number += 10
        counter.append(iteration_number)
        loss.append(epoch_loss)
                    
        # save model and result every 10 epochs
        if epoch % 10 == 0:
            save_figures(iteration_number,counter,loss,final_path,epoch,epoch_loss,curr_lr)
            #save model state up to this epoch
            if epoch_loss < min_loss:
                min_loss = epoch_loss
                torch.save(model.state_dict(), os.path.join(final_path,"epoch{}_loss{}_lr{}.pt".format(epoch,epoch_loss,curr_lr)))
    return model

def save_figures(iteration_number,counter,loss,final_path,epoch,epoch_loss,curr_lr):
    plt.plot(counter,loss)
    plt.xlabel('Epoch (10:1 scale)')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(final_path,"epoch{}_loss{}_lr{}.png".format(epoch,epoch_loss,curr_lr)))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.train()
print("Starting training")
model = train()
# torch.save(model.state_dict(), "model_sequential_isGoodMaybe2_{}.pt".format())
print("Model Saved Successfully") 