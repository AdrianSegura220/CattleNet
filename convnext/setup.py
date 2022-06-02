from __future__ import print_function
from __future__ import division
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
from model_test_original import test_thresholds
from custom_dataset_bce import CustomImageDataset_Validation, CustomImageDatasetBCE
import wandb
from custom_dataset import CustomImageDataset
from torch.utils.data import DataLoader
from cattleNetTest import CattleNet
from tqdm import tqdm
from model_test_original import test


# save or not model snapshots
save_models = True

# save or not figures
save_figs = False

# wandb setup (logging progress to online platform)
use_wandb = True

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
step_lr = 10
lr=1e-3
in_channel = 3
batch_size = 8
num_epochs = 40
n_shot = 15
k_folds = 8
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
                "Avg. avg_f1-score d=0.6": validation_results['avg_f1-score'][4]
            })

        # 0.1,0.25,0.4,0.5,0.6

        print('Epoch avg precision: {}'.format(validation_results['avg_precision']))
        print('Epoch avg recall: {}'.format(validation_results['avg_recall']))
        print('Epoch avg balanced accuracy: {}'.format(validation_results['avg_balanced_acc']))
        print('Epoch avg f-score: {}'.format(validation_results['avg_f1-score']))

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
        criterion = ContrastiveLoss()

        params = model.parameters()
        # setup optimizer (use Adam technique to optimize parameters (GD with momentum and RMS prop))
        # by default: learning rate = 0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0
        optimizer = optim.Adam(params,lr=lr) 
        scheduler = StepLR(optimizer, step_size=step_lr, gamma=0.1)

        dataset_training = CustomImageDatasetBCE(img_dir='../../dataset/Raw/Combined/',transform=transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),transforms.Resize((240,240))]),annotations_csv='./training_testing_folds/training_annotations_fold{}.csv'.format(i))
        dataset_validation = CustomImageDatasetBCE(img_dir='../../dataset/Raw/Combined/',transform=transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),transforms.Resize((240,240))]),annotations_csv='./training_testing_folds/validation_annotations_fold{}.csv'.format(i))
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