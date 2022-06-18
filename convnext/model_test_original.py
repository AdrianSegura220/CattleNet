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
import sklearn.metrics as metrics
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time
import os
import random
import copy
from torchvision.io import read_image
# from convnext.custom_dataset_bce import CustomImageDatasetBCE
from custom_dataset_bce import CustomImageDataset_Validation, CustomImageDatasetBCE, OneShotImageDataset                                                                                                                                                                            
from torch.utils.data import DataLoader
from cattleNetTest_v3 import CattleNetV3
from tqdm import tqdm

def compute_roc_auc(out1,out2,labels,batch,epoch):
    cos = nn.CosineSimilarity(dim=1,eps=1e-6)
    scores = cos(out1,out2)
    fpr, tpr, thresholds = metrics.roc_curve(labels.cpu().numpy(), scores.cpu().numpy())
    roc_auc = metrics.auc(fpr, tpr)
    plt.gca().cla()
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    for i in range(0,len(thresholds)):
        plt.axvline(x=fpr[i])
        plt.text(x=fpr[i]+0.02,y=tpr[i]-0.1,s=str(thresholds[i]))
    plt.savefig('../roc_figures/roc_batch{}__EPOCHnr{}.png'.format(batch,epoch))
    
    print(len(thresholds))
    print(len(fpr))
    print(len(tpr))

    bestThreshold = thresholds[np.argmax(tpr-fpr)]

    print(bestThreshold)

    return roc_auc,bestThreshold

"""
    remark: use CustomImageDatasetBCE for this task
"""
def test_thresholds(test_dataset: CustomImageDatasetBCE, model_directory: str = '', model_version: str = '',model = None,is_load_model = False,thresholds = [0.5],epoch=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total = 0
    correct = 0
    results = []
    stats = [{} for i in range(0,len(thresholds))]
    data_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    avg_precision = [0.0 for i in range(0,len(thresholds))]
    avg_recall = [0.0 for i in range(0,len(thresholds))]
    avg_balanced_acc = [0.0 for i in range(0,len(thresholds))]
    avg_fscore = [0.0 for i in range(0,len(thresholds))]
    avg_false_positive_rate = [0.0 for i in range(0,len(thresholds))]
    avg_precision_to_reduce = [0 for i in range(0,len(thresholds))]
    avg_recall_to_reduce = [0 for i in range(0,len(thresholds))]
    avg_balancedacc_to_reduce = [0 for i in range(0,len(thresholds))]
    avg_fscore_to_reduce = [0 for i in range(0,len(thresholds))]
    avg_false_positive_rate_to_reduce = [0 for i in range(0,len(thresholds))]
    avg_tp = [0 for i in range(0,len(thresholds))]
    avg_tn = [0 for i in range(0,len(thresholds))]
    avg_fp = [0 for i in range(0,len(thresholds))]
    avg_fn = [0 for i in range(0,len(thresholds))]
    zero_recall = [0 for i in range(0,len(thresholds))]
    zero_precision = [0 for i in range(0,len(thresholds))]
    zero_acc = [0 for i in range(0,len(thresholds))]
    accuracy = 0
    avg_auc = 0.0
    avg_best_threshold = 0.0

    if is_load_model:
        pass
        # data_dict = encode_dataset(test_dataset, model_directory, model_version)
    else:
        batches = 0
        for data in data_loader:
            batches += 1
            anchor = data[0].to(device)
            images = data[1].to(device)
            labels = data[2].to(device)

            # forward pass using anchor and images
            anchor_res,images_res = model(anchor,images)

            auc_result, best_threshold = compute_roc_auc(anchor_res,images_res,labels,batches,epoch)

            # add calculated values to running sum to average at the end
            avg_auc += auc_result
            avg_best_threshold += best_threshold



        return avg_auc/batches,avg_best_threshold/batches
        # UNCOMMENT
        #     distances_sq = torch.sub(anchor_res,images_res).pow(2).sum(1)
            
            
        #     """
        #         Iterate through each threshold and save stats
        #     """
        #     for i,d in enumerate(thresholds):
        #         classifications = (distances_sq < d).float() # use broadcasting to discern for each difference whether it is smaller than d => mark it as same image 
        #         temp_result = (classifications == labels).float() # for each element, decide whether they are match the actual labels
        #         true_positives = sum([1 if (l == 1 and classifications[i] == 1) else 0 for i,l in enumerate(labels)])
        #         true_negatives = sum([1 if (l == 0 and classifications[i] == 0) else 0 for i,l in enumerate(labels)])
        #         false_positives = sum([1 if (l == 0 and classifications[i] == 1) else 0 for i,l in enumerate(labels)])
        #         false_negatives = sum([1 if (l == 1 and classifications[i] == 0) else 0 for i,l in enumerate(labels)])
                
        #         avg_tp[i] += true_positives
        #         avg_tn[i] += true_negatives
        #         avg_fp[i] += false_positives
        #         avg_fn[i] += false_negatives


        #         if true_positives + false_positives > 0:
        #             precision = true_positives/(true_positives + false_positives)
        #         else:
        #             precision = -1
        #             zero_precision[i] += 1
                
        #         if true_positives + false_negatives > 0:
        #             recall = true_positives/(true_positives + false_negatives)
        #         else:
        #             recall = -1
        #             zero_recall[i] += 1

        #         if false_positives + true_negatives > 0:
        #             true_negative_rate = true_negatives/(false_positives+true_negatives)
        #         else:
        #             true_negative_rate = -1

        #         if recall != -1 and true_negative_rate != -1:
        #             balanced_acc = (recall+true_negative_rate)/2
        #         else:
        #             balanced_acc = -1
        #             zero_acc[i] += 1

        #         if recall != -1 and precision != -1 and recall > 0.0001 and precision > 0.0001:
        #             fscore = 2*recall*precision/(precision+recall)
        #         else:
        #             fscore = -1

        #         if recall != -1 and false_positives + true_positives > 0:
        #             false_positive_rate = false_positives/(false_positives + true_negatives)
        #         else:
        #             false_positive_rate = -1

        #         # accuracy = temp_result.sum(1)/classifications.size()[0]
        #         stats[i] = {
        #             'precision': precision,
        #             'recall': recall,
        #             'balanced_accuracy': balanced_acc,
        #             'f1-score': fscore,
        #             'false_positive_rate': false_positive_rate
        #         }

        #     for i,s in enumerate(stats): # so for each distance threhold recorded result add the value and at the end divide by total no. batches
                
        #         """
        #             So, for each distance threshold being tested, if the measurement was invalid, then simply
        #             not add value because it is not helpful, otherwise add the obtained value for that indicator, 
        #             but just make sure that the final division of accumulated indicators for the mean is made
        #             for only the values added and not for those that were invalid
        #         """
        #         if s['precision'] != -1:
        #             avg_precision[i] += s['precision']
        #         else:
        #             avg_precision_to_reduce[i] += 1

        #         if s['recall'] != -1:
        #             avg_recall[i] += s['recall']
        #         else:
        #             avg_recall_to_reduce[i] += 1

        #         if s['balanced_accuracy'] != -1:
        #             avg_balanced_acc[i] += s['balanced_accuracy']
        #         else:
        #             avg_balancedacc_to_reduce[i] += 1

        #         if s['f1-score'] != -1:
        #             avg_fscore[i] += s['f1-score']
        #         else:
        #             avg_fscore_to_reduce[i] += 1

        #         if s['false_positive_rate'] != -1:
        #             avg_false_positive_rate[i] += s['false_positive_rate']
        #         else:
        #             avg_false_positive_rate_to_reduce[i] += 1

        # """
        #     for each accumulated statistic for each distance threshold, divide by the amount of batches to calculate
        #     the average precision, recall and balanced_acc using such distance threshold
        # """
        # for i in range(0,len(thresholds)):
        #     avg_precision[i] /= (batches-avg_precision_to_reduce[i])
        #     avg_recall[i] /= (batches-avg_recall_to_reduce[i])
        #     avg_balanced_acc[i] /= (batches-avg_balancedacc_to_reduce[i])
        #     avg_fscore[i] /= (batches-avg_fscore_to_reduce[i])
        #     avg_false_positive_rate[i] /= (batches-avg_false_positive_rate_to_reduce[i])

        
        #     avg_tp[i] /= batches
        #     avg_tn[i] /= batches
        #     avg_fp[i] /= batches
        #     avg_fn[i] /= batches

        # """
        #     Compute roc with average values
        # """
        

        # # print('Avg tp: {}'.format(avg_tp))
        # # print('Avg tn: {}'.format(avg_tn))
        # # print('Avg fp: {}'.format(avg_fp))
        # # print('Avg fn: {}'.format(avg_fn))
        # print('Zero precision: {}'.format(zero_precision))
        # print('Zero recall: {}'.format(zero_recall))
        # print('Zero acc: {}'.format(zero_acc))

        # """
        #     return results in form of a dictionary containing avg values for precision, recall and balanced accuracy for
        #     each of the thresholds. So effectively, each of those values is an array containing each an average result per
        #     distance threshold tested
        # """
        # return {
        #     'avg_precision': avg_precision,
        #     'avg_recall': avg_recall,
        #     'avg_balanced_acc': avg_balanced_acc,
        #     'avg_f1-score': avg_fscore,
        #     'avg_fpr': avg_false_positive_rate
        # }

"""
    In this function, the one-shot classification capacity of the model is tested.
    To do this, we first calculate the embeddings for all the dataset, store them in a
    dictionary for each class. Then for each of the classes, we select one of its images at random
    as anchor for the test, we then select an image of all classes (including the same of the anchor,
    but a different image). Once we do this, we use our defined distance threshold.
"""
def one_shot_test(test_dataset: OneShotImageDataset,model,threshold,use_argmin,quantify_wrong):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loader = DataLoader(test_dataset,batch_size=1)
    correct = 0
    incorrect = 0
    images = {}

    # check what data[1] is

    # generate all the embeddings and store them
    for data in data_loader:
        if int(data[1]) not in images:
            images[int(data[1])] = []
            
        img = data[0].to(device)
        out,dummy = model(img,img)
        images[int(data[1])].append(out)

    for j,k in enumerate(images.keys()):
        if len(images[k]) > 1:
            anchor_idx = random.randint(0,len(images[k])-1) # select an index of anchor cow label
            anchor = images[k][anchor_idx].to(device) # select anchor
            rest = torch.Tensor(len(images.keys()),4096).to(device) # allocate space for all cow classes available
            for i,k2 in enumerate(images.keys()):
                idx = random.randint(0,len(images[k2])-1) # some random idx for current class
                if i == j: # if we are selecting an image for the same cow as anchor, make sure the image is not the same
                    # select positive example
                    idx = random.randint(0,len(images[k])-1)
                    while idx == anchor_idx:
                        idx = random.randint(0,len(images[k])-1) # assign a positive example image that is not the same as the anchor
                    
                rest[i] = images[k2][idx].to(device) # assign selected image from specific label
            
            # rest
            # we have to subtract the anchor from the large tensor e.g. rest-anchor to use advantage of broadcasting
            # differences = torch.sub(rest,anchor).pow(2).sum(1)
            cos = nn.CosineSimilarity(dim=1,eps=1e-6)
            differences = cos(rest,anchor)
            results = (differences < threshold).float()
            if use_argmin:
                # selected = torch.argmin(differences)
                selected = torch.argmax(differences)
                if selected == j:
                    print('C')
                    correct += 1
                else:
                    incorrect += 1
                    print('I')
            else:
                if results[j] == 1.0 and results.sum(0) == 1:
                    correct += 1
                else:
                    if quantify_wrong:
                        print('False positives (falsely taken as same img): ',results.sum(0)-1)
                    incorrect += 1
        else:
            continue

    return correct/(correct+incorrect)
    

"""
    Description:
        Go through each image, do forward pass through CNN to generate feature vectors
        and pair them with their labels in a dict.
        For each label, revise whether own images have the smallest euclidean distance
        compared to other images
        This method is quite inefficient, considering to do vector quantization
"""
def test(test_dataset: CustomImageDataset_Validation,n, model_directory: str = '', model_version: str = '',model = None,is_load_model = False):
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