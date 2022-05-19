from __future__ import print_function
from __future__ import division
from base64 import encode
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
from torchvision.io import read_image
from custom_dataset import CustomImageDataset
from torch.utils.data import DataLoader
from cattleNetTest import CattleNet
from tqdm import tqdm

"""
    Description:
        This function finds embeddings for all images in test data and
        build a dictionary to be used for accuracy testing in the test method.

    Parameters:
        test_dataset: dataset that returns test images and labels
        model_directory: directory containing the saved models from training
        model_version: name of the specific model to be tested (and loaded) 
"""
def encode_dataset(test_dataset: CustomImageDataset, model_directory: str, model_version: str, model=None,is_load_model=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if is_load_model:
        folder_plus_modelV = os.path.join(model_directory,model_version)
        model = CattleNet()
        model.eval()
        model.load_state_dict(torch.load(folder_plus_modelV)) # load model that is to be tested
        model.to(device)
    else:
        model.eval()

    data_dict = {}

    data_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    for data in data_loader:
        imgs1 = data[0].to(device)
        imgs2 = data[1].to(device)
        labels1 = data[2].to(device)
        labels2 = data[3].to(device)
        out1,out2 = model(imgs1,imgs2)

        if labels1 not in data_dict:
            data_dict[labels1] = []
        
        if labels2 not in data_dict:
            data_dict[labels2] = []
        
        data_dict[labels1].append(out1)
        data_dict[labels2].append(out2)

    return data_dict


"""
TO TRY:
    - calculate an average of each cow's representation vectors and use that as the
    vector for comparisons

    Description:
        Go through each image, do forward pass through CNN to generate feature vectors
        and pair them with their labels in a dict.
        For each label, revise whether own images have the smallest euclidean distance
        compared to other images
        This method is quite inefficient, considering to do vector quantization
"""
def test(test_dataset: CustomImageDataset, model_directory: str = '', model_version: str = '',model: CattleNet = None,is_load_model = True):
    correct = 0
    wrong = 0
    euclidean = torch.nn.PairwiseDistance(2) # setup pairwise distance with parameter 2, to specify (sum of squares)^(1/n) => (sum of squares)^(1/2)
    if is_load_model:
        data_dict = encode_dataset(test_dataset, model_directory, model_version)
    else:
        data_dict = encode_dataset(test_dataset, model_directory, model_version,model,is_load_model)
    

    results = {}

    """
        Basically save the label of cow's embedding with closest similarity per embedding for each cow.
        Then compare with own labels and if the minimum distance at the end comes from an embedding of
        same cow, then the embeddings are being correctly set by network.
    """
    for key in data_dict.keys(): # for each label (cow)
        results[key] = []
        for embedding in data_dict[key]:
            results[key].append((float('inf'),'dummy_label'))
            for key2 in data_dict.keys():
                if key != key2:
                    for embedding2 in data_dict[key2]:
                        pwdistance = euclidean(embedding,embedding2)
                        if pwdistance < results[key][-1]: # if current embedding of different cow (embedding2) has a smaller distance, set to new min
                            results[key][-1][0] = (pwdistance,key2)
                        
    for key in data_dict.keys(): # for each label (cow)
        for i,embedding in enumerate(data_dict[key]):
            for j,embedding2 in enumerate(data_dict[key]):
                if i != j:
                    pwdistance = euclidean(embedding,embedding2)
                    if pwdistance < results[key][i][0]:
                        correct += 1
                    else:
                        wrong += 1
    
    # return accuracy
    return correct/(correct+wrong)
                        


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#../../BachelorsProject/Trainings/bs8_adam_lrem3_05_17/
# model = CattleNet()
# model.load_state_dict(torch.load('../../BachelorsProject/Trainings/model_InitialLR0.001_lrDecay1wStep10_trainSize1066_testSize267_datetime18-5H14M48/epoch42_loss0.2555221329206851_lr1.0000000000000002e-07.pt'))
# model.to(device)
# model.eval()

# dataset = CustomImageDataset(dataset_folder='../../dataset/Raw/RGB (320 x 240)/',img_dir='../../dataset/Raw/Combined/',transform=transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]))

"""
    INFORMAL CHECK
"""
# imgt1 = read_image('../../dataset/Raw/Combined/9884_6_1.jpg').float()/255.0
# imgte1 = read_image('../../dataset/Raw/Combined/9884_4_2.jpg').float()/255.0

# imgt2 = read_image('../../dataset/Raw/Combined/9912_7_1.jpg').float()/255.0
# imgte2 = read_image('../../dataset/Raw/Combined/9912_9_1.jpg').float()/255.0

# informal_transform=transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

# imgt1 = informal_transform(imgt1)
# imgte1 = informal_transform(imgte1)

# imgt2 = informal_transform(imgt2)
# imgte2 = informal_transform(imgte2)

# imgt1 = imgt1[None,:,:,:].cuda()
# imgte1 = imgte1[None,:,:,:].cuda()
# imgt2 = imgt2[None,:,:,:].cuda()
# imgte2 = imgte2[None,:,:,:].cuda()

# print(imgt1.size())
# exit()

# print(device)
# imgt1.to(device)
# imgte1.to(device)
# imgt2.to(device)
# imgte2.to(device)

# out_1,out_e1 = model(imgt1,imgte1)
# out_2,out_e2 = model(imgt2,imgte2)
# euclid = torch.nn.PairwiseDistance(2)
# print('equal images (first pair)',euclid(out_1,out_e1))
# print('dissimilar images (imgt1 and imgt2)',euclid(out_1,out_e2))
# print('equal images (second pair)',euclid(out_2,out_e2))
# print('dissimilar images (imgte2 and imgte1)',euclid(out_e2,out_e1))
# exit()
"""
    END INFORMAL CHECK
"""


# setup training and testing sets
# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# for data in data_loader:
#     # print(data[0].size())
#     # exit()
#     imgs1 = data[0].to(device)
#     imgs2 = data[1].to(device)
#     labels1 = data[2].to(device)
#     labels2 = data[3].to(device)
#     out1,out2 = model(imgs1,imgs2)
#     print('img1: ',data[2],' img2: ',data[3])
#     euclidean = torch.nn.PairwiseDistance(2)
#     output = euclidean(out1,out2)
#     print('euclidean distance: ',output)
#     print('')
