from __future__ import print_function
from __future__ import division
from base64 import encode
from turtle import forward
from sklearn.metrics import balanced_accuracy_score
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
from convnext.custom_dataset_bce import CustomImageDatasetBCE
from custom_dataset_bce import CustomImageDataset_Validation
from torch.utils.data import DataLoader
from cattleNetTest_v3 import CattleNetV3
from tqdm import tqdm

"""
    remark: use CustomImageDatasetBCE for this task
"""
def test_thresholds(test_dataset: CustomImageDatasetBCE,n, model_directory: str = '', model_version: str = '',model: CattleNetV3 = None,is_load_model = False,thresholds = [0.5]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total = 0
    correct = 0
    results = []
    stats = [[] for i in range(0,thresholds)]
    data_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    avg_precision = 0
    avg_recall = 0
    avg_balanced_acc = 0
    accuracy = 0

    if is_load_model:
        pass
        # data_dict = encode_dataset(test_dataset, model_directory, model_version)
    else:
        batches = 0
        for data in data_loader:
            batches += 1
            anchor = data[0].to(device)
            images = data[1][0].to(device)
            labels = data[2][0]

            # forward pass using anchor and images
            anchor_res,images_res = model(anchor,images)

            distances_sq = torch.sub(anchor_res,images_res).pow(2).sum(1)
            
            """
                Iterate through each threshold and save stats
            """
            for d in thresholds:
                classifications = (distances_sq < d).float() # use broadcasting to discern for each difference whether it is smaller than d => mark it as same image 
                temp_result = (classifications == labels).float() # for each element, decide whether they are match the actual labels
                true_positives = sum([1 if (l == 1 and classifications[i] == 1) else 0 for i,l in enumerate(labels)])
                true_negatives = sum([1 if (l == 0 and classifications[i] == 0) else 0 for i,l in enumerate(labels)])
                false_positives = sum([1 if (l == 1 and classifications[i] == 0) else 0 for i,l in enumerate(labels)])
                false_negatives = sum([1 if (l == 0 and classifications[i] == 1) else 0 for i,l in enumerate(labels)])
                precision = true_positives/(true_positives + false_positives)
                recall = true_positives/(true_positives + false_negatives)
                true_negative_rate = true_negatives/(false_positives+true_negatives)
                balanced_acc = (recall+true_negative_rate)/2
                # accuracy = temp_result.sum(1)/classifications.size()[0]
                stats.append({
                    'precision': precision,
                    'recall': recall,
                    'balanced_accuracy': balanced_acc
                })
            for s in stats:
                avg_precision += s['precision']
                avg_recall += s['recall']
                avg_balanced_acc += s['balanced_accuracy']
        
        avg_precision /= batches
        avg_recall /= batches
        avg_balanced_acc /= batches

        return {
            'avg_precision': avg_precision, 
            'avg_recall': avg_recall,
            'avg_balanced_acc': avg_balanced_acc
        }


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
            anchor_res,images_res = model(anchor,images)
            
            correct_idx = torch.argmax(data[2])
            max_elem = torch.argmin(torch.sub(anchor_res,images_res).pow(2).sum(1))
            if max_elem == correct_idx:
                # print('results :',res)
                # print('reference: ', data[2])
                correct += 1
            total += 1
    
    # return accuracy
    print('correct: {}/{}'.format(correct,total))
    return correct/total