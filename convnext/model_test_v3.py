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
from custom_dataset_bce import CustomImageDataset_Validation
from torch.utils.data import DataLoader
from cattleNetTest_v3 import CattleNetV3
from tqdm import tqdm


"""
    Description:
        Go through each image, do forward pass through CNN to generate feature vectors
        and pair them with their labels in a dict.
        For each label, revise whether own images have the smallest euclidean distance
        compared to other images
        This method is quite inefficient, considering to do vector quantization
"""
def test(test_dataset: CustomImageDataset_Validation,n, model_directory: str = '', model_version: str = '',model: CattleNetV3 = None,is_load_model = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total = 0
    correct = 0
    data_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    if is_load_model:
        pass
        # data_dict = encode_dataset(test_dataset, model_directory, model_version)
    else:
        for data in data_loader:
            anchor = data[0].repeat(n,1,1,1).to(device)
            images = data[1][0].to(device)
            labels = data[2][0]
            # forward pass using anchor and images
            res = model(anchor,images)
            
            correct_idx = torch.argmax(data[2])
            max_elem = torch.argmax(res)
            if max_elem == correct_idx:
                # print('results :',res)
                # print('reference: ', data[2])
                correct += 1
            total += 1
    
    # return accuracy
    print('correct: {}/{}'.format(correct,total))
    return correct/total