from __future__ import print_function
from __future__ import division
import itertools
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
from custom_dataset_v2 import CustomImageDatasetV2
from torch.utils.data import DataLoader
from cattleNetTest_v2 import CattleNetV2
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
step_lr = 1
lr=1e-3
in_channel = 3
batch_size = 8
num_epochs = 40


# wandb.config = {
#   "learning_rate": lr,
#   "epochs": num_epochs,
#   "batch_size": batch_size
# }

if not loadtest:
    # instantiate SNN
    model = CattleNetV2()
    model.train()
    model.to(device)

    # loss function
    criterion = ContrastiveLoss()

    params = model.parameters()
    # setup optimizer (use Adam technique to optimize parameters (GD with momentum and RMS prop))
    # by default: learning rate = 0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0
    optimizer = optim.Adam(params) 
    scheduler = StepLR(optimizer, step_size=step_lr, gamma=0.99)
    print()
    dataset = CustomImageDataset(dataset_folder='../../dataset/Raw/RGB (320 x 240)/',img_dir='../../dataset/Raw/Combined/',transform=transforms.Compose([transforms.Resize((240,240))]))
    test_dataset = CustomImageDataset(dataset_folder='../../dataset/Raw/RGB (320 x 240)/',img_dir='../../dataset/Raw/Combined/',transform=transforms.Compose([transforms.Resize((240,240))]))

# setup learning rate scheduler

# setup training and testing sets
# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    len(dataset)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


def load_and_test(fname):
    print('here')
    model = CattleNetV2()
    model.load_state_dict(torch.load(fname)) # load model that is to be tested
    model.eval()
    model.to(device)
    test_dataset = CustomImageDataset(dataset_folder='../../dataset/Raw/RGB (320 x 240)/',img_dir='../../dataset/Raw/Combined/',transform=transforms.Compose([transforms.Resize((240,240))]))
    # model.eval()
    with torch.no_grad():
        acc = test(test_dataset,model=model,is_load_model=False)
    print('Accuracy: ', acc)

def train():
    min_loss = 99999999999999.0
    loss = []
    counter = []
    iteration_number = 0
    epoch_loss = 0.0
    iterations_loop = 0
    plot_pairs = False
    # create directory for current training results
    final_path = os.path.join(path_to_results,'CNV2Comb_lr{}'.format(lr))
    # os.mkdir(final_path)
    figures = []
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

            # out1 and out2 contain the embeddings for all image pairs

            if plot_pairs:
                listimgs = [(img,labels1[i],imgs1[i].to('cpu')) for i,img in enumerate(out1)]
                listimgs = listimgs + [(img,labels2[i],imgs2[i].to('cpu')) for i,img in enumerate(out2)]
            else:
                listimgs = [(img,labels1[i]) for i,img in enumerate(out1)]
                listimgs = listimgs + [(img,labels2[i]) for i,img in enumerate(out2)]


            combinations = list(itertools.combinations(listimgs,2)) # generate all combinations of embedding pairs 
            images_combinations = torch.Tensor(len(combinations),2,out1.size()[1]) # create new tensor with dimensions of output vectors to store all combinations of embeddings
            are_equal = torch.Tensor(len(combinations))

            
                  
            # generate tensor labels for all combinations
            # basically generate a tensor of shape: (#Combinations of two elements,2,# embedding length)
            # e.g. given 8 images: (28,2,4096)
            for i,combination in enumerate(combinations):
                if plot_pairs:
                    figures.append(plt.figure(figsize=(10, 7)))
                    figures[i].add_subplot(2, 2, 1)
                    plt.imshow(combination[0][2].permute(1,2,0))
                    plt.axis('off')
                    plt.title("{}".format(combination[0][1]))
                    figures[i].add_subplot(2, 2, 2)
                    plt.imshow(combination[1][2].permute(1,2,0))
                    plt.title("{}".format(combination[1][1]))
                are_equal[i] = (combination[0][1] == combination[1][1]).float()
                images_combinations[i][0] = combination[0][0] # set left image to left image of combination
                images_combinations[i][1] = combination[1][0] # set right image to right image of combination
            
            # print(are_equal)
            # plt.show()
            # exit()

            # basically images_combinations[:,0] contains all left images of pairs and images_combinations[:,1] contains all right images of pairs
            # then are_equal contains whether each of those pairs are of the same label or not
            loss_contrastive = criterion(images_combinations[:,0],images_combinations[:,1],are_equal)
            loss_contrastive.backward()
            optimizer.step()
            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(loss=loss_contrastive.item())
            epoch_loss += loss_contrastive.item()
            iterations_loop += 1
        
        
        model.eval()
        with torch.no_grad():
            acc = test(test_dataset,model=model,is_load_model=False)
        model.train()
        print('Accuracy: ', acc)
        scheduler.step()
        epoch_loss /= iterations_loop
        curr_lr = optimizer.state_dict()['param_groups'][0]['lr']

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
    
    return model

def save_figures(iteration_number,counter,loss,final_path,epoch,epoch_loss,curr_lr):
    plt.plot(counter,loss)
    plt.xlabel('Epoch (10:1 scale)')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(final_path,"epoch{}_loss{}_lr{}.png".format(epoch,epoch_loss,curr_lr)))

if loadtest:
    load_and_test('../../BachelorsProject/Trainings/CNV2Comb_lr0.001/epoch50_loss0.4793714602550347_lr0.0006050060671375363.pt')
else:
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Starting training")
    fmodel = train()
    torch.save(fmodel.state_dict(), "CNV2_final.pt")
    # set model to eval mode
    # model.eval()
    # test_dataset = CustomImageDatasetV2(dataset_folder='../../dataset/Raw/RGB (320 x 240)/',img_dir='../../dataset/Raw/Combined/',transform=transforms.Compose([transforms.Resize((240,240))]),testing=True)
    # acc = test(test_dataset,model=model,is_load_model=False)
    # print('Accuracy: {}'.format(acc))
    # wandb.log({"accuracy": acc})
    print("Model Saved Successfully") 