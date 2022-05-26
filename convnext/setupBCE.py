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
from custom_dataset_bce import CustomImageDatasetBCE,CustomImageDataset_Validation
from torch.utils.data import DataLoader
from cattleNetTest_v3 import CattleNetV3
from tqdm import tqdm
from model_test_v3 import test


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
step_lr = 1
lr=6e-3
in_channel = 3
batch_size = 32
num_epochs = 200


wandb.config = {
  "learning_rate": lr,
  "epochs": num_epochs,
  "batch_size": batch_size
}

if not loadtest:
    # instantiate SNN
    model = CattleNetV3(freezeLayers=False)
    model.to(device)

    # loss function
    # criterion = ContrastiveLoss()
    criterion = nn.BCELoss()

    params = model.parameters()
    # setup optimizer (use Adam technique to optimize parameters (GD with momentum and RMS prop))
    # by default: betas=(0.9, 0.999), eps=1e-08, weight_decay=0
    optimizer = optim.Adam(params,lr=lr) 
    scheduler = StepLR(optimizer, step_size=step_lr, gamma=0.99) # anneal lr by 1% of previous lr each epoch
train_dataset = CustomImageDatasetBCE(img_dir='../../dataset/Raw/Combined/',transform=transforms.Compose([transforms.Resize((240,240))]))
test_dataset = CustomImageDataset_Validation(img_dir='../../dataset/Raw/Combined/',n=8,transform=transforms.Compose([transforms.Resize((240,240))]))
# train_dataset = CustomImageDatasetBCE(img_dir='../../dataset/Raw/TrainingDivided/Training/',transform=transforms.Compose([transforms.Resize((240,240))]))
# test_dataset = CustomImageDataset_Validation(img_dir='../../dataset/Raw/TrainingDivided/Validation/',n=8,transform=transforms.Compose([transforms.Resize((240,240))]))

data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def load_and_test(fname):
    print('here')
    model = CattleNetV3()
    model.load_state_dict(torch.load(fname)) # load model that is to be tested
    model.eval()
    model.to(device)
    model.eval()
    acc = test(test_dataset,model=model,is_load_model=False)
    print(acc)


def train():
    min_loss = 99999999999999.0
    loss = []
    accuracy = []
    counter = []
    iteration_number = 0
    epoch_loss = 0.0
    iterations_loop = 0
    # create directory for current training results
    final_path = os.path.join(path_to_results,'CattleNetV3_WValidation_lr{}_BCE_datetime{}-{}H{}M{}S{}'.format(lr,datetime.datetime.today().day,datetime.datetime.today().month,datetime.datetime.today().hour,datetime.datetime.today().minute,datetime.datetime.today().second))
    # os.mkdir(final_path)
    last_epoch = 0
    for epoch in range(1,num_epochs):
        loop = tqdm(data_loader,leave=False,total=len(data_loader))
        epoch_loss = 0.0
        iterations_loop = 0
        for data in loop:
            optimizer.zero_grad()
            imgs1 = data[0].to(device)
            imgs2 = data[1].to(device)

            labels = torch.reshape(data[2],(data[2].size()[0],1)).float().to(device)
            res = model(imgs1,imgs2)
            # print('YHAT: ',res)
            # print('Y: ', labels)
            loss_contrastive = criterion(res,labels)
            loss_contrastive.backward()
            optimizer.step()
            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(loss=loss_contrastive.item())
            epoch_loss += loss_contrastive.item()
            iterations_loop += 1
        scheduler.step()
        epoch_loss /= iterations_loop
        curr_lr = optimizer.state_dict()['param_groups'][0]['lr']


        # validation:
        model.eval()
        with torch.no_grad():
            epoch_acc = test(test_dataset,8,model=model,is_load_model=False)
        model.train()

        #print details of elapsed epoch
        print("lr {}".format(curr_lr))
        print("Epoch {}\n Current loss {}\n Current Accuracy {}\n".format(epoch,epoch_loss,epoch_acc))
        # wandb.log({"loss": epoch_loss})

        # maintain epochs in scales of 10
        iteration_number += 10
        counter.append(iteration_number)
        loss.append(epoch_loss)
        accuracy.append(epoch_acc)

        # save model and result every 10 epochs
        if epoch % 10 == 0:
            save_figures(iteration_number,counter,loss,final_path,epoch,epoch_loss,curr_lr,accuracy,epoch_acc)
            #save model state up to this epoch
            if epoch_loss < min_loss:
                min_loss = epoch_loss
                torch.save(model.state_dict(), os.path.join(final_path,"epoch{}_loss{}_lr{}.pt".format(epoch,epoch_loss,curr_lr)))
        last_epoch = epoch
    
    torch.save(model.state_dict(), os.path.join(final_path,"CNV3_FinalModel_epoch{}.pt".format(last_epoch)))
    print("Model Saved Successfully") 
    # set model to eval mode
    # model.eval()
    # with torch.no_grad():
    #     acc = test(test_dataset,model=model,is_load_model=False)
    #     print('Accuracy: {}'.format(acc))
    # wandb.log({"accuracy": acc})
    
    return model

def save_figures(iteration_number,counter,loss,final_path,epoch,epoch_loss,curr_lr,accuracy,epoch_acc):
    plt.plot(counter,loss)
    plt.xlabel('Epoch (10:1 scale)')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(final_path,"LOSS{}_epoch{}_lr{}.png".format(epoch_loss,epoch,curr_lr)))
    plt.plot(counter,accuracy)
    plt.xlabel('Epoch (10:1 scale)')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(final_path,"ACC{}_epoch{}_lr{}.png".format(epoch_acc,epoch_loss,curr_lr)))

if loadtest:
    load_and_test('../../BachelorsProject/Trainings/model_InitialLR0.001_lrDecay1wStep10_trainSize1066_testSize267_datetime20-5H2M11/epoch30_loss0.2583860134455695_lr1.0000000000000002e-06.pt')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    print("Starting training")
    fmodel = train()