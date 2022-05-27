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
from custom_dataset_bce import CustomImageDataset_Validation, CustomImageDatasetBCE
import wandb
from custom_dataset import CustomImageDataset
from torch.utils.data import DataLoader
from cattleNetTest import CattleNet
from tqdm import tqdm
from model_test import test


# wandb setup (logging progress to online platform)
# wandb.init(project="cattleNet-arch1", entity="adriansegura220")


# load and test a model version (no training)
loadtest = False

# setup device type
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# results folders
path_to_results = '../../BachelorsProject/Trainings/'   

#hyperparams
lrDecay = 1
step_lr = 10
lr=1e-3
in_channel = 3
batch_size = 8
num_epochs = 40
n_shot = 15



wandb.config = {
  "learning_rate": lr,
  "epochs": num_epochs,
  "batch_size": batch_size
}

if not loadtest:
    # instantiate SNN
    model = CattleNet(freezeLayers=False)
    model.to(device)

    # loss function
    criterion = ContrastiveLoss()

    params = model.parameters()
    # setup optimizer (use Adam technique to optimize parameters (GD with momentum and RMS prop))
    # by default: learning rate = 0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0
    optimizer = optim.Adam(params,lr=lr) 
    scheduler = StepLR(optimizer, step_size=step_lr, gamma=0.1)

dataset = CustomImageDatasetBCE(img_dir='../../dataset/Raw/Combined/',transform=transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]))
validation = CustomImageDataset_Validation(n=n_shot,img_dir='../../dataset/Raw/Combined/',transform=transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]))
# setup learning rate scheduler

# # setup training and testing sets
# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
# len(dataset)

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


def load_and_test(fname):
    model = CattleNet()
    model.load_state_dict(torch.load(fname)) # load model that is to be tested
    model.eval()
    model.to(device)
    model.eval()
    acc = test(validation,model=model,is_load_model=False)
    print(acc)


def train():
    min_loss = 99999999999999.0
    loss = []
    counter = []
    iteration_number = 0
    epoch_loss = 0.0
    iterations_loop = 0
    # create directory for current training results
    final_path = os.path.join(path_to_results,'CattleNetContrastive_lr{}_BCE_datetime{}-{}H{}M{}S{}'.format(lr,datetime.datetime.today().day,datetime.datetime.today().month,datetime.datetime.today().hour,datetime.datetime.today().minute,datetime.datetime.today().second))
    # os.mkdir(final_path)

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
            labels = data[2].to(device)
            out1,out2 = model(imgs1,imgs2)
            loss_contrastive = criterion(out1,out2,labels)
            loss_contrastive.backward()
            optimizer.step()
            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(loss=loss_contrastive.item())
            epoch_loss += loss_contrastive.item()
            iterations_loop += 1
        scheduler.step()
        epoch_loss /= iterations_loop
        curr_lr = optimizer.state_dict()['param_groups'][0]['lr']

        # if epoch == 14:
        #     torch.cuda.empty_cache()
        #     model.unfreeze_layers()

        #print details of elapsed epoch
        print("lr {}".format(curr_lr))
        print("Epoch {}\n Current loss {}\n".format(epoch,epoch_loss))
        # wandb.log({"loss": epoch_loss})

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
    
        # set model to eval mode
        # validation:
        model.eval()
        with torch.no_grad():
            epoch_acc = test(validation,n=n_shot,model=model,is_load_model=False)
        model.train()
        print('Epoch Accuracy: {}'.format(epoch_acc))
    
    return model

def save_figures(iteration_number,counter,loss,final_path,epoch,epoch_loss,curr_lr):
    plt.plot(counter,loss)
    plt.xlabel('Epoch (10:1 scale)')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(final_path,"epoch{}_loss{}_lr{}.png".format(epoch,epoch_loss,curr_lr)))

if loadtest:
    load_and_test('../../BachelorsProject/Trainings/model_InitialLR0.001_lrDecay1wStep10_trainSize1066_testSize267_datetime20-5H2M11/epoch30_loss0.2583860134455695_lr1.0000000000000002e-06.pt')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    print("Starting training")
    model = train()
    # torch.save(model.state_dict(), "model_sequential_isGoodMaybe2_{}.pt".format())
    print("Model Saved Successfully") 