from __future__ import print_function
from __future__ import division
import sys


sys.path.insert(0,'..')
from turtle import forward
from torchvision import datasets, models, transforms 
from torchvision import datasets, transforms as T
from contrastive_loss import ContrastiveLoss
from torch.optim.lr_scheduler import StepLR
import sklearn.metrics as metrics
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


# save or not model snapshots
save_models = True

# save or not figures
save_figs = False

# wandb setup (logging progress to online platform)
use_wandb = True

# load and test a model version (no training)
loadtest = False

# setup device type
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# results folders
path_to_results = '../../BachelorsProject/Trainings/'   

#hyperparams
lrDecay = 1
step_lr = 1
lr=45e-5
in_channel = 3
batch_size = 128
num_epochs = 200
n_shot = 15
k_folds = 8
thresholds_to_test = [0.1,0.25,0.4,0.5,0.6]

def compute_roc_auc(out1,out2,labels,epoch):
    cos = nn.CosineSimilarity(dim=1,eps=1e-6)
    scores = cos(out1,out2)
    fpr, tpr, thresholds = metrics.roc_curve(labels.cpu().numpy(), scores.cpu().numpy())
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('../roc_figures/roc_epoch{}.png'.format(epoch))
    plt.clf()


def train(d_loader,dataset_validation,dataset_validation_training):
    min_loss = 99999999999999.0
    max_auc = -1.0
    max_one_shot = -1.0
    loss = []
    # arrays to save the best values for model training
    precision = [0.0 for i in range(0,len(thresholds_to_test))]
    recall = [0.0 for i in range(0,len(thresholds_to_test))]
    balanced_acc = [0.0 for i in range(0,len(thresholds_to_test))]
    f_score = [0.0 for i in range(0,len(thresholds_to_test))]
    avg_best_threshold = 0.0
    avg_best_training_threshold = 0.0
    avg_auc_validation_testing = 0.0
    avg_auc_validation_training = 0.0
    avg_loss_training_validation = 0.0
    avg_loss_testing_validation = 0.0

    accuracy = []
    epoch_acc = 0.0
    counter = []
    iteration_number = 0
    epoch_loss = 0.0
    iterations_loop = 0
    # create directory for current training results
    if save_models or save_figs:
        final_path = os.path.join(path_to_results,'FINALAUC_Contrastive{}_datetime{}-{}H{}M{}S{}'.format(lr,datetime.datetime.today().day,datetime.datetime.today().month,datetime.datetime.today().hour,datetime.datetime.today().minute,datetime.datetime.today().second))
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
            validation_results,avg_best_calculated_threshold,loss_testing_validation = test_thresholds(dataset_validation,thresholds=thresholds_to_test,model=model,epoch=epoch,mode='testing',criterion=criterion)
            validation_results_training,avg_best_calculated_training_threshold,loss_training_validation = test_thresholds(dataset_validation_training,thresholds=thresholds_to_test,model=model,epoch=epoch,mode='training',criterion=criterion)
            # validation_training_results = test_thresholds(dataset_validation_training,thresholds=thresholds_to_test,model=model)
            one_shot = one_shot_test(dataset_one_shot,model,0.5,True,True)

            avg_best_threshold += avg_best_calculated_threshold # add last best-calculated threshold to running sum
            avg_best_training_threshold += avg_best_calculated_training_threshold # add last best-calculated threshold to running sum
            avg_auc_validation_testing += validation_results # add to divide in the end to get whole model's average
            avg_auc_validation_training += validation_results_training # add to divide in the end to get whole model's average
            avg_loss_training_validation += loss_training_validation
            avg_loss_testing_validation += loss_testing_validation
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
                "Avg. AUC value per epoch for testing validation": validation_results,
                "Avg. AUC value per epoch for training validation": validation_results_training,
                "Avg. one-shot performance": one_shot,
                "loss": epoch_loss,
                "Best threshold running average for testing validation: ": avg_best_threshold/epoch,
                "Best threshold running average for training validation": avg_best_training_threshold/epoch,
                "Avg. loss testing validation": avg_loss_testing_validation/epoch,
                "Avg. loss training validation": avg_loss_training_validation/epoch
            })
            #UNCOMMENT
            # wandb.log({
            #     "loss": epoch_loss,
            #     "Avg. balanced accuracy d=0.1": validation_results['avg_balanced_acc'][0],
            #     "Avg. balanced accuracy d=0.25": validation_results['avg_balanced_acc'][1],
            #     "Avg. balanced accuracy d=0.4": validation_results['avg_balanced_acc'][2],
            #     "Avg. balanced accuracy d=0.5": validation_results['avg_balanced_acc'][3],
            #     "Avg. balanced accuracy d=0.6": validation_results['avg_balanced_acc'][4],
            #     "Avg. precision d=0.1": validation_results['avg_precision'][0],
            #     "Avg. precision d=0.25": validation_results['avg_precision'][1],
            #     "Avg. precision d=0.4": validation_results['avg_precision'][2],
            #     "Avg. precision d=0.5": validation_results['avg_precision'][3],
            #     "Avg. precision d=0.6": validation_results['avg_precision'][4],
            #     "Avg. recall d=0.1": validation_results['avg_recall'][0],
            #     "Avg. recall d=0.25": validation_results['avg_recall'][1],
            #     "Avg. recall d=0.4": validation_results['avg_recall'][2],
            #     "Avg. recall d=0.5": validation_results['avg_recall'][3],
            #     "Avg. recall d=0.6": validation_results['avg_recall'][4],
            #     "Avg. avg_f1-score d=0.1": validation_results['avg_f1-score'][0],
            #     "Avg. avg_f1-score d=0.25": validation_results['avg_f1-score'][1],
            #     "Avg. avg_f1-score d=0.4": validation_results['avg_f1-score'][2],
            #     "Avg. avg_f1-score d=0.5": validation_results['avg_f1-score'][3],
            #     "Avg. avg_f1-score d=0.6": validation_results['avg_f1-score'][4],
            #     "One-shot score": one_shot,
            #     "Avg. balanced accuracy d=0.1": validation_training_results['avg_balanced_acc'][0],
            #     "Avg. balanced accuracy d=0.25": validation_training_results['avg_balanced_acc'][1],
            #     "Avg. balanced accuracy d=0.4": validation_training_results['avg_balanced_acc'][2],
            #     "Avg. balanced accuracy d=0.5": validation_training_results['avg_balanced_acc'][3],
            #     "Avg. balanced accuracy d=0.6": validation_training_results['avg_balanced_acc'][4],
            #     "Avg. precision d=0.1": validation_training_results['avg_precision'][0],
            #     "Avg. precision d=0.25": validation_training_results['avg_precision'][1],
            #     "Avg. precision d=0.4": validation_training_results['avg_precision'][2],
            #     "Avg. precision d=0.5": validation_training_results['avg_precision'][3],
            #     "Avg. precision d=0.6": validation_training_results['avg_precision'][4],
            #     "Avg. recall d=0.1": validation_training_results['avg_recall'][0],
            #     "Avg. recall d=0.25": validation_training_results['avg_recall'][1],
            #     "Avg. recall d=0.4": validation_training_results['avg_recall'][2],
            #     "Avg. recall d=0.5": validation_training_results['avg_recall'][3],
            #     "Avg. recall d=0.6": validation_training_results['avg_recall'][4],
            #     "Avg. avg_f1-score d=0.1": validation_training_results['avg_f1-score'][0],
            #     "Avg. avg_f1-score d=0.25": validation_training_results['avg_f1-score'][1],
            #     "Avg. avg_f1-score d=0.4": validation_training_results['avg_f1-score'][2],
            #     "Avg. avg_f1-score d=0.5": validation_training_results['avg_f1-score'][3],
            #     "Avg. avg_f1-score d=0.6": validation_training_results['avg_f1-score'][4],
            # })

        
        # UNCOMMENT
        # print('Epoch avg precision: {}'.format(validation_results['avg_precision']))
        # print('Epoch avg recall: {}'.format(validation_results['avg_recall']))
        # print('Epoch avg balanced accuracy: {}'.format(validation_results['avg_balanced_acc']))
        # print('Epoch avg f-score: {}'.format(validation_results['avg_f1-score']))
        print('Epoch avg. auc value: {}'.format(validation_results))
        print('Epoch one-shot accuracy: {}'.format(one_shot))
        print('Best threshold running-average value: {}'.format(avg_best_threshold/epoch))

        """
            Add obtained statistic, in order to average it at the very end, also
            save the best performing statistic for each. Use an exponential moving
            average to give bigger weight to latest results, because initial epochs
            might make the average seem low because of low performing starting values.
            NOTE: implement F1 score too
        """
        # for d in range(0,len(thresholds_to_test)):
        #     precision[d] = validation_results['avg_precision'][d]
        #     recall[d] = validation_results['avg_recall'][d]
        #     # if there is a new best max value, then update these
        #     # we do not update precision and recall in that way, because separately they are not that meaningfull
        #     balanced_acc[d] = validation_results['avg_balanced_acc'][d] if balanced_acc[d] < validation_results['avg_balanced_acc'][d] else balanced_acc[d]
        #     f_score[d] = validation_results['avg_f1-score'][d] if f_score[d] < validation_results['avg_f1-score'][d] else f_score[d]

        # save model and result every 10 epochs
        if epoch % 10 == 0:
            if save_figs:
                save_figures(iteration_number,counter,loss,final_path,epoch,epoch_loss,curr_lr,accuracy,epoch_acc)
            
            #save model state up to this epoch
        if save_models and (validation_results > max_auc or one_shot > max_one_shot):
            max_one_shot = one_shot
            max_auc = validation_results
            torch.save(model.state_dict(), os.path.join(final_path,"epoch{}_AUC{}_oneshot{}.pt".format(epoch,validation_results,one_shot)))
    
    # return model and the best values for balanced accuracy and also for f-score
    avg_auc_validation_testing /= num_epochs # divide to get total average
    avg_auc_validation_training /= num_epochs # divide to get toal average

    return model,avg_auc_validation_testing,avg_auc_validation_training

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
    avg_folds_auc_testing_validation = 0.0
    avg_folds_auc_training_validation = 0.0
    # balanced_acc = [0.0 for i in range(0,len(thresholds_to_test))]
    # f_score = [0.0 for i in range(0,len(thresholds_to_test))]

    for i in range(0,k_folds):
        # instantiate SNN model
        model = CattleNet(freezeLayers=True)
        model.to(device)
        # loss function
        criterion = ContrastiveLoss()

        if use_wandb:
            run = wandb.init(project="cattleNet-arch1", entity="adriansegura220",reinit=True)

        if use_wandb:
            wandb.config = {
            "learning_rate": lr,
            "epochs": num_epochs,
            "batch_size": batch_size
            }

        params = model.parameters()
        # setup optimizer (use Adam technique to optimize parameters (GD with momentum and RMS prop))
        # by default: learning rate = 0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0
        optimizer = optim.Adam(params,lr=lr)
        scheduler = StepLR(optimizer, step_size=step_lr, gamma=0.99)

        dataset_training = CustomImageDatasetBCE(img_dir='../../dataset/CORF3D_combined/',transform=transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]),annotations_csv='./training_testing_folds_corf3d/training_annotations_fold{}.csv'.format(i),isCorf3d=True)
        dataset_validation = CustomImageDatasetBCE(img_dir='../../dataset/CORF3D_combined/',transform=transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]),annotations_csv='./training_testing_folds_corf3d/validation_annotations_fold{}.csv'.format(i),isCorf3d=True)
        dataset_validation_training = CustomImageDatasetBCE(img_dir='../../dataset/CORF3D_combined/',transform=transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]),annotations_csv='./training_testing_folds_corf3d/training_validation_annotations_fold{}.csv'.format(i),isCorf3d=True)
        dataset_one_shot = OneShotImageDataset(img_dir='../../dataset/CORF3D_combined/',transform=transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]),annotations_csv='./training_testing_folds_corf3d/validation_annotations_fold{}.csv'.format(i),isCorf3d=True)
        data_loader = DataLoader(dataset_training, batch_size=batch_size, shuffle=True)


        model.train()
        print("Starting training")
        model,avg_auc_validation_testing,avg_auc_validation_training = train(d_loader=data_loader,dataset_validation=dataset_validation,dataset_validation_training=dataset_validation_training)
        

        if use_wandb:
            wandb.log({
                "Total avg. auc testing validation of the fold": avg_auc_validation_testing,
                "Total avg. auc training validation of the fold": avg_auc_validation_training
            })
        
        avg_folds_auc_testing_validation += avg_auc_validation_testing
        avg_folds_auc_training_validation += avg_auc_validation_training
        run.finish()

        # for j in range(0,len(thresholds_to_test)):
        #     # add to compute average at the end
        #     balanced_acc[j] += res_balanced_acc[j]
        #     f_score[j] += res_f_score[j]
        #     # if use_wandb:
        #     #     wandb.log({
        #     #         "Best result balanced accuracy for d={}".format(thresholds_to_test[j]): res_balanced_acc[j],
        #     #         "Best result f-score for d={}".format(thresholds_to_test[j]): res_f_score[j]
        #     #     })
    print('Average auc for testing validation amongst all folds:',avg_folds_auc_testing_validation/k_folds)
    print('Average auc for training validation amongst all folds:',avg_folds_auc_training_validation/k_folds)
    # argmx_acc = 0
    # max_acc = 0
    # argmx_fscore = 0
    # max_fscore = 0

    """
        Calculate averages of all folds and select indices of distances with highest
        balanced accuracy and highest f-scores
    """
    # for i in range(0,len(thresholds_to_test)):
    #     balanced_acc[i] /= k_folds
    #     if balanced_acc[i] > max_acc:
    #         max_acc = balanced_acc[i]
    #         argmx_acc = i
        
    #     f_score[i] /= k_folds
    #     if f_score[i] > max_fscore:
    #         max_fscore = f_score[i]
    #         argmx_fscore = i

    # print("Best distance threshold based on balanced acc {}-folds: d = {}".format(k_folds,thresholds_to_test[argmx_acc]))
    # print("Best distance threshold based on F1-score {}-folds: d = {}".format(k_folds,thresholds_to_test[argmx_fscore]))

    # if use_wandb:
    #     wandb.log({
    #         "Best distance threshold based on balanced acc {}-folds".format(k_folds): thresholds_to_test[argmx_acc],
    #         "Best distance threshold based on f1-score {}-folds".format(k_folds): thresholds_to_test[argmx_fscore]
    #     })
        
        # 0.1,0.25,0.4,0.5,0.6
        
    # torch.save(model.state_dict(), "model_sequential_isGoodMaybe2_{}.pt".format())