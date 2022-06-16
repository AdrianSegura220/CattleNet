from __future__ import print_function
from __future__ import division
from math import dist
import sys


sys.path.insert(0,'..')
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
from model_test_original import test_thresholds,one_shot_test
from custom_dataset_bce import CustomImageDataset_Validation, CustomImageDatasetBCE, OneShotImageDataset
import wandb
from custom_dataset import CustomImageDataset
from torch.utils.data import DataLoader
from cattleNetTest import CattleNet
from tqdm import tqdm
from model_test_original import test
from torch.utils.data import default_collate
from dataset_triplets import SimpleTriplet
from triplet_loss import TripletLoss


# save or not model snapshots
save_models = False

# save or not figures
save_figs = False

# wandb setup (logging progress to online platform)
use_wandb = False

if use_wandb:
    wandb.init(project="cattleNet-arch1", entity="adriansegura220")


# load and test a model version (no training)
loadtest = False

# setup device type
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# results folders
path_to_results = '../../BachelorsProject/Trainings/'   

#hyperparams
lrDecay = 1
step_lr = 1
lr=15e-4
in_channel = 3
batch_size = 16
num_epochs = 150
n_shot = 15
k_folds = 1
thresholds_to_test = [0.1,0.25,0.4,0.5,0.6]

if use_wandb:
    wandb.config = {
    "learning_rate": lr,
    "epochs": num_epochs,
    "batch_size": batch_size
    }

# def load_and_test(fname):
#     model = CattleNet()
#     model.load_state_dict(torch.load(fname)) # load model that is to be tested
#     model.eval()
#     model.to(device)
#     model.eval()
#     acc = test(validation,model=model,is_load_model=False)
#     print(acc)


def mineHardTriples(images,labels,model):
    output = torch.Tensor()
    model.eval()
    with torch.no_grad():
        output = model(images)

    # compute pairwise distances
    dot_embeddings = torch.matmul(output,torch.transpose(output,0,1))

    diagonal = torch.diagonal(dot_embeddings) # obtain diagonal (contains the squared norm of each embedding)

    # compute pairwise distance (squared distance) matrix:
    distances = diagonal[None,:] - 2*dot_embeddings + diagonal[:,None]

    labelsMatrix = torch.Tensor(distances.size())

    mask = torch.zeros(distances.size()).bool()

    hard_positives = torch.Tensor(distances.size()[0])

    cnt = 0
    for i in range(0,distances.size()[0]):
        for j in range(0,distances.size()[0]):
            if i != j and labels[i] == labels[j]: # to enforce images are not the same but that they have the same label
                # print(i,j)
                # print(output[i][:5])
                # print()
                # print(output[j][:5])
                if mask[j,i] == False:
                    mask[i,j] = True
                    cnt += 1
    # print(cnt)

    for i in range(0,distances.size()[0]):
        # currRow = torch.masked_select(distances[i],mask[i])
        max_dist = -1.0
        max_idx = -1
        for j in range(0,distances.size()[0]):
            if mask[i,j] == True and max_dist < distances[i][j]:
                max_idx = j
        hard_positives[i] = max_idx if max_idx != -1 else -1
        
    # mask_minus_one = hard_positives.not_equal(-1)
    # hard_positives = torch.masked_select(hard_positives,mask_minus_one)

    # declare tensor for hard_negatives
    hard_negatives = torch.Tensor(distances.size()[0])

    for i in range(0,distances.size()[0]):
        for j in range(0,distances.size()[0]):
            mask[i,j] = True if labels[i] != labels[j] else False # in order to enforce only different labels from anchor

    for i in range(0,distances.size()[0]):
        min_dist = 999999.0
        min_idx = -1
        for j in range(0,distances.size()[0]):
            if mask[i,j] == True and min_dist > distances[i][j]:
                min_idx = j

        # currRow = torch.masked_select(distances[i],mask[i])
        # hard_negatives[i] = torch.argmin(currRow) if currRow.size()[0] > 0 else -1
        hard_negatives[i] = min_idx if min_idx != -1 else -1
        
    triplets = []
    for i in range(0,distances.size()[0]):
        if hard_positives[i] != -1.0 and hard_negatives[i] != -1.0:
            triplets.append((i,hard_positives[i],hard_negatives[i])) # push a tuple having indices for images in current batch: (i,j,k), where label(images[i]) == label(images[j]) but they are not the same image and the hardest one to identify. Also label(images[i]) != label(images[k]) but they are the closest images between the anchor and negative examples 

    anchors = torch.Tensor(len(triplets))
    positives = torch.Tensor(len(triplets))
    negatives = torch.Tensor(len(triplets))

    for i in range(0,len(triplets)):
        anchors[i] = triplets[i][0]
        positives[i] = triplets[i][1]
        negatives[i] = triplets[i][2]

    model.train() # set model back to train mode
    return anchors.int(),positives.int(),negatives.int() # return the indices of images for each (anchors, positives, negatives in order)



def train(d_loader,dataset_validation):
    min_loss = 99999999999999.0
    loss = []
    # arrays to save the best values for model training
    precision = [0.0 for i in range(0,len(thresholds_to_test))]
    recall = [0.0 for i in range(0,len(thresholds_to_test))]
    balanced_acc = [0.0 for i in range(0,len(thresholds_to_test))]
    f_score = [0.0 for i in range(0,len(thresholds_to_test))]

    accuracy = []
    epoch_acc = 0.0
    counter = []
    iteration_number = 0
    epoch_loss = 0.0
    iterations_loop = 0
    # create directory for current training results
    if save_models or save_figs:
        final_path = os.path.join(path_to_results,'CattleNetContrastive_lr{}_BCE_datetime{}-{}H{}M{}S{}'.format(lr,datetime.datetime.today().day,datetime.datetime.today().month,datetime.datetime.today().hour,datetime.datetime.today().minute,datetime.datetime.today().second))
        os.mkdir(final_path)

    for epoch in range(1,num_epochs):
        loop = tqdm(d_loader,leave=False,total=len(d_loader))
        epoch_loss = 0.0
        iterations_loop = 0
        for data in loop:
            # print(data[0].size())
            label = 0
            #### do something with images ...
            optimizer.zero_grad()


            imgs = data[0].to(device)
            labels = data[1]
            anchorIndices, positiveIndices, negativeIndices = mineHardTriples(imgs,labels,model)

            if anchorIndices.size()[0] == 0:
                continue

            anchors = torch.index_select(imgs,0,anchorIndices)
            positives = torch.index_select(imgs,0,positiveIndices)
            negatives = torch.index_select(imgs,0,negativeIndices)

            for i in range(0,anchorIndices.size()[0]):
                print(labels[anchorIndices[i]],labels[positiveIndices[i]],labels[negativeIndices[i]])   

            # print(anchors.size())
            # print(positives.size())
            # print(negatives.size())

            # exit()

            # generate embeddings for all triplets to then compare them using triplet loss
            anchorsEmbeddings = model(anchors)
            positivesEmbeddings = model(positives)
            negativesEmbeddings = model(negatives)

            # pass embeddings generated for triplets to triplet loss
            lossFunction = criterion(anchorsEmbeddings,positivesEmbeddings,negativesEmbeddings)
            lossFunction.backward()
            optimizer.step()
            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(loss=lossFunction.item())
            epoch_loss += lossFunction.item()
            iterations_loop += 1
        scheduler.step()
        epoch_loss /= iterations_loop
        curr_lr = optimizer.state_dict()['param_groups'][0]['lr']

        #print details of elapsed epoch
        print("lr {}".format(curr_lr))
        print("Epoch {}\n Current loss {}\n".format(epoch,epoch_loss))

        # maintain epochs in scales of 10
        iteration_number += 10
        counter.append(iteration_number)
        loss.append(epoch_loss)

        # validation:
        model.eval()
        with torch.no_grad():
            # epoch_acc = test(dataset_validation,n=n_shot,model=model,is_load_model=False)
            validation_results = test_thresholds(dataset_validation,thresholds=thresholds_to_test,model=model)
            one_shot = one_shot_test(dataset_one_shot,model,0.5,True,True)
            """
                validation results returns an array with results for each distance threshold
                e.g. given 3 thresholds to test: [0.1,0.3,0.5], then for each statistic (precision,recall and balanced acc)
                we will have 3 results, one for each threshold, because each threshold will give potentially different
                results on how well the equal and different embeddings can be discriminated
            """
        model.train()

        """
            Improve this, remove hardcoded threshold indexing, make it dynamic
        """
        if use_wandb:
            wandb.log({
                "loss": epoch_loss,
                "Avg. balanced accuracy d=0.1": validation_results['avg_balanced_acc'][0],
                "Avg. balanced accuracy d=0.25": validation_results['avg_balanced_acc'][1],
                "Avg. balanced accuracy d=0.4": validation_results['avg_balanced_acc'][2],
                "Avg. balanced accuracy d=0.5": validation_results['avg_balanced_acc'][3],
                "Avg. balanced accuracy d=0.6": validation_results['avg_balanced_acc'][4],
                "Avg. precision d=0.1": validation_results['avg_precision'][0],
                "Avg. precision d=0.25": validation_results['avg_precision'][1],
                "Avg. precision d=0.4": validation_results['avg_precision'][2],
                "Avg. precision d=0.5": validation_results['avg_precision'][3],
                "Avg. precision d=0.6": validation_results['avg_precision'][4],
                "Avg. recall d=0.1": validation_results['avg_recall'][0],
                "Avg. recall d=0.25": validation_results['avg_recall'][1],
                "Avg. recall d=0.4": validation_results['avg_recall'][2],
                "Avg. recall d=0.5": validation_results['avg_recall'][3],
                "Avg. recall d=0.6": validation_results['avg_recall'][4],
                "Avg. avg_f1-score d=0.1": validation_results['avg_f1-score'][0],
                "Avg. avg_f1-score d=0.25": validation_results['avg_f1-score'][1],
                "Avg. avg_f1-score d=0.4": validation_results['avg_f1-score'][2],
                "Avg. avg_f1-score d=0.5": validation_results['avg_f1-score'][3],
                "Avg. avg_f1-score d=0.6": validation_results['avg_f1-score'][4],
                "One-shot score": one_shot
            })

        # 0.1,0.25,0.4,0.5,0.6

        print('Epoch avg precision: {}'.format(validation_results['avg_precision']))
        print('Epoch avg recall: {}'.format(validation_results['avg_recall']))
        print('Epoch avg balanced accuracy: {}'.format(validation_results['avg_balanced_acc']))
        print('Epoch avg f-score: {}'.format(validation_results['avg_f1-score']))
        print('Epoch one-shot accuracy: {}'.format(one_shot))

        """
            Add obtained statistic, in order to average it at the very end, also
            save the best performing statistic for each. Use an exponential moving
            average to give bigger weight to latest results, because initial epochs
            might make the average seem low because of low performing starting values.
            NOTE: implement F1 score too
        """
        for d in range(0,len(thresholds_to_test)):
            precision[d] = validation_results['avg_precision'][d]
            recall[d] = validation_results['avg_recall'][d]
            # if there is a new best max value, then update these
            # we do not update precision and recall in that way, because separately they are not that meaningfull
            balanced_acc[d] = validation_results['avg_balanced_acc'][d] if balanced_acc[d] < validation_results['avg_balanced_acc'][d] else balanced_acc[d]
            f_score[d] = validation_results['avg_f1-score'][d] if f_score[d] < validation_results['avg_f1-score'][d] else f_score[d]

        # save model and result every 10 epochs
        if epoch % 10 == 0:
            if save_figs:
                save_figures(iteration_number,counter,loss,final_path,epoch,epoch_loss,curr_lr,accuracy,epoch_acc)
            
            #save model state up to this epoch
            if save_models and epoch_loss < min_loss:
                min_loss = epoch_loss
                torch.save(model.state_dict(), os.path.join(final_path,"epoch{}_loss{}_lr{}_maxAvgBalancedAcc{}.pt".format(epoch,epoch_loss,curr_lr,max(validation_results['avg_balanced_acc']))))
    
    # return model and the best values for balanced accuracy and also for f-score
    return model,balanced_acc,f_score

def save_figures(iteration_number,counter,loss,final_path,epoch,epoch_loss,curr_lr,accuracy,epoch_acc):
    plt.plot(counter,loss)
    plt.xlabel('Epoch (10:1 scale)')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(final_path,"LOSS{}_epoch{}_lr{}.png".format(epoch_loss,epoch,curr_lr)))
    plt.plot(counter,accuracy)
    plt.xlabel('Epoch (10:1 scale)')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(final_path,"ACC{}_epoch{}_lr{}.png".format(epoch_acc,epoch_loss,curr_lr)))

def custom_collate(batch):
    batch = list(filter(lambda img: img is not None,batch))
    return default_collate(batch)

if loadtest:
    pass
    # load_and_test('../../BachelorsProject/Trainings/model_InitialLR0.001_lrDecay1wStep10_trainSize1066_testSize267_datetime20-5H2M11/epoch30_loss0.2583860134455695_lr1.0000000000000002e-06.pt')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # final_path = os.path.join(path_to_results,'CattleNetContrastive_folds{}_lr{}_BCE_datetime{}-{}H{}M{}S{}'.format(k_folds,lr,datetime.datetime.today().day,datetime.datetime.today().month,datetime.datetime.today().hour,datetime.datetime.today().minute,datetime.datetime.today().second))
    # os.mkdir(final_path)

    # create arrays to keep track of these performance values for each threshold tested
    balanced_acc = [0.0 for i in range(0,len(thresholds_to_test))]
    f_score = [0.0 for i in range(0,len(thresholds_to_test))]

    for i in range(0,k_folds):
        # instantiate SNN model
        model = CattleNet(freezeLayers=True)
        model.to(device)
        # loss function
        # criterion = ContrastiveLoss()
        criterion = TripletLoss()

        params = model.parameters()
        # setup optimizer (use Adam technique to optimize parameters (GD with momentum and RMS prop))
        # by default: learning rate = 0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0
        optimizer = optim.Adam(params,lr=lr) 
        scheduler = StepLR(optimizer, step_size=step_lr, gamma=0.99)

        # dataset_training = CustomImageDatasetBCE(img_dir='../../dataset/Preprocessed/Combined/',transform=transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]),annotations_csv='./training_testing_folds/training_annotations_fold{}.csv'.format(i))
        dataset_training = SimpleTriplet(img_dir='../../dataset/Preprocessed/Combined/',transform=transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]),annotations_csv='./training_testing_folds/validation_annotations_fold{}.csv'.format(i))
        dataset_validation = CustomImageDatasetBCE(img_dir='../../dataset/Preprocessed/Combined/',transform=transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]),annotations_csv='./training_testing_folds/validation_annotations_fold{}.csv'.format(i))
        dataset_one_shot = OneShotImageDataset(img_dir='../../dataset/Preprocessed/Combined/',transform=transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]),annotations_csv='./training_testing_folds/validation_annotations_fold{}.csv'.format(i))
        data_loader = DataLoader(dataset_training, batch_size=batch_size, shuffle=True)


        model.train()
        print("Starting training")
        model,res_balanced_acc,res_f_score = train(d_loader=data_loader,dataset_validation=dataset_validation)

        for j in range(0,len(thresholds_to_test)):
            # add to compute average at the end
            balanced_acc[j] += res_balanced_acc[j]
            f_score[j] += res_f_score[j]
            if use_wandb:
                wandb.log({
                    "Best result balanced accuracy for d={}".format(thresholds_to_test[j]): res_balanced_acc[j],
                    "Best result f-score for d={}".format(thresholds_to_test[j]): res_f_score[j]
                })
        
    argmx_acc = 0
    max_acc = 0
    argmx_fscore = 0
    max_fscore = 0

    """
        Calculate averages of all folds and select indices of distances with highest
        balanced accuracy and highest f-scores
    """
    for i in range(0,len(thresholds_to_test)):
        balanced_acc[i] /= k_folds
        if balanced_acc[i] > max_acc:
            max_acc = balanced_acc[i]
            argmx_acc = i
        
        f_score[i] /= k_folds
        if f_score[i] > max_fscore:
            max_fscore = f_score[i]
            argmx_fscore = i

    print("Best distance threshold based on balanced acc {}-folds: d = {}".format(k_folds,thresholds_to_test[argmx_acc]))
    print("Best distance threshold based on F1-score {}-folds: d = {}".format(k_folds,thresholds_to_test[argmx_fscore]))

    if use_wandb:
        wandb.log({
            "Best distance threshold based on balanced acc {}-folds".format(k_folds): thresholds_to_test[argmx_acc],
            "Best distance threshold based on f1-score {}-folds".format(k_folds): thresholds_to_test[argmx_fscore]
        })
        
        # 0.1,0.25,0.4,0.5,0.6
        
    # torch.save(model.state_dict(), "model_sequential_isGoodMaybe2_{}.pt".format())
    print("Model Saved Successfully") 