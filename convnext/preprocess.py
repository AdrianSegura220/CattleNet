"""
    Author: Adrian Segura Lorente
    Description: This file serves the purpose of generating some important information                                                                
    related to the raw dataset to enable the establishment of a dataset
    and afterwards training.
"""

import os
from numpy import number
import pandas as pd
import sys
sys.path.insert(0,'..')

def checkSimilarities():
    for i in range(0,8):
        training_df = pd.read_csv('./training_testing_folds/training_annotations_fold{}.csv'.format(i))
        validation_df = pd.read_csv('./training_testing_folds/validation_annotations_fold{}.csv'.format(i))
        tr_set = set([elem for elem in training_df['Label']])
        vl_set = set([elem for elem in validation_df['Label']])
        res = tr_set.intersection(vl_set)
        print('INTERSECTION FOR FOLD {}:\n'.format(i),res)

def generate_folds(directory:str, k: int,isCorf3d: int):
    try: # case directory does not exist
        if isCorf3d:
            os.mkdir('./training_testing_folds_corf3d')
        else:
            os.mkdir('./training_testing_folds')
    except FileExistsError:
        pass # else do nothing and just overwrite files

    files = os.listdir(directory)
    files.sort()
    files = files[1:] # first element is never a valid file but a .ds file, so skip it
    # print(files)
    labels = [file.split('_')[0] for file in files] # basically, only take label, as files are structured as: <label number>_<some other data>.jpeg
    
    number_files = len(files)-1

    validation_size = int(number_files/k)
    train_size = number_files-validation_size
    train_splits = [{'train': [],'labels':[]} for i in range(0,k)]

    # process:
    # generate list of all files and from those, use a loop to generate each fold and generate csv files
    for i in range(0,k):
        train_current = []
        validation_current = []

        upper_limit_validation = validation_size*i+validation_size if validation_size*i+validation_size <= number_files else number_files
        
        # generate validation data frames for current split
        validation_current = files[validation_size*i:upper_limit_validation] # select only the specific portion for the validation at fold k
        labels_validation_current = [file.split('_')[0] for file in validation_current] # generate labels for k'th validation fold

        # save training fold (will be modified later to extract the training validation examples, so not converting yet to a csv file)
        training_current = files[0:validation_size*i]+files[upper_limit_validation:] # select only the portion for the training at fold k
        train_splits[i]['train'] += training_current
        labels_training_current = [file.split('_')[0] for file in training_current] # generate labels for k'th training fold
        train_splits[i]['labels'] += labels_training_current
        
        labelsToFilter = []
        for j in range(0,len(labels_validation_current)):
            if labels_validation_current[j] in labels_training_current:
                labelsToFilter.append(labels_validation_current[j])

        validation_current = list(filter(lambda x: x.split('_')[0] not in labelsToFilter,validation_current))
        labels_validation_current = list(filter(lambda x: x not in labelsToFilter,labels_validation_current))

        df = pd.DataFrame({'Path': validation_current,'Label': labels_validation_current}) # generate data frame having as columns the path to an image and labels for each of such images
        if isCorf3d:
            df.to_csv('./training_testing_folds_corf3d/validation_annotations_fold{}.csv'.format(i))
        else:
            df.to_csv('./training_testing_folds/validation_annotations_fold{}.csv'.format(i))
        # df = pd.DataFrame({'Path': training_current,'Label': labels_training_current}) # generate data frame having as columns the path to an image and labels for each of such images
        #df.to_csv('./training_testing_folds/training_annotations_fold{}.csv'.format(i))

    # generate annotations for all folds for training validation
    prevLabel = '0'
    for i in range(0,k):
        train_validation = [] # to put all 
        labels_train_validation = []
        for j in range(0,len(train_splits[i]['train'])):
            if len(train_validation) < validation_size: # this guard is kept in order to have approximately equally-sized testing validation and training validation sets
                if j < len(train_splits[i]['labels']) and train_splits[i]['labels'][j] != prevLabel:
                    prevLabel = train_splits[i]['labels'][j]
                    current_label = train_splits[i]['labels'][j]
                    if j+3 < len(train_splits[i]['train']) and train_splits[i]['labels'][j+3] == train_splits[i]['labels'][j]:
                        # add elements to train validation set
                        train_validation += train_splits[i]['train'][j:j+2] # grab two elements if there are more than 3 examples (i.e. 4 >=) (so that we can have at least a positive example for each case (training and training validation))
                        labels_train_validation += [current_label for k in range(0,2)] # add label for each of the images added to the training-validation set
                        # re-arrange training set for i ' th fold
                        train_splits[i]['train'] = train_splits[i]['train'][:j]+train_splits[i]['train'][j+2:] # modify training set to exclude those images selected for training - validation set
                        train_splits[i]['labels'] = train_splits[i]['labels'][:j]+train_splits[i]['labels'][j+2:]
            else:
                break # this would mean that we reached at least the size of the testing validation set, so we leave the rest of the training set as is and use the composed training validation set for the current fold with the added images

        if isCorf3d:
            df = pd.DataFrame({'Path': train_splits[i]['train'],'Label': train_splits[i]['labels']}) # generate data frame having as columns the path to an image and labels for each of such images
            df.to_csv('./training_testing_folds_corf3d/training_annotations_fold{}.csv'.format(i))

            df = pd.DataFrame({'Path': train_validation,'Label': labels_train_validation}) # generate data frame having as columns the path to an image and labels for each of such images
            df.to_csv('./training_testing_folds_corf3d/training_validation_annotations_fold{}.csv'.format(i))
        else:
            df = pd.DataFrame({'Path': train_splits[i]['train'],'Label': train_splits[i]['labels']}) # generate data frame having as columns the path to an image and labels for each of such images
            df.to_csv('./training_testing_folds/training_annotations_fold{}.csv'.format(i))

            df = pd.DataFrame({'Path': train_validation,'Label': labels_train_validation}) # generate data frame having as columns the path to an image and labels for each of such images
            df.to_csv('./training_testing_folds/training_validation_annotations_fold{}.csv'.format(i))

def generate_annotations_direct(directory: str,name_output: str):
    files = os.listdir(directory)
    files.sort()
    files = files[1:]  # first element is never a valid file but a .ds file, so skip it
    # print(files)
    labels = [file.split('_')[0] for file in files]
    df = pd.DataFrame({'Path': files,'Label': labels}) # generate data frame having as columns the path to an image and labels for each of such images
    df.to_csv('{}.csv'.format(name_output))

"""
    This function generates data frame having an image path at index 1
    and an image label (e.g. which cow it belongs to) at index 2
"""
def generate_annotations(directory: str,name_of_file: str = 'annotations') -> None:
    dirnames = os.listdir(directory)
    dirnames.sort() # order by filename
    dirnames = dirnames[1:]
    indices = []
    labels = []
    for dname in dirnames:
        indices = indices + os.listdir(directory+'/'+dname+'/') # list all file names inside directory dname (where dname is the label of each cow. e.g. 0089)
        labels = labels + [dname for lb in os.listdir(directory+'/'+dname+'/')] # for each image in each directory append label
    df = pd.DataFrame({'Path': indices,'Label': labels}) # generate data frame having as columns the path to an image and labels for each of such images
    df.to_csv('{}.csv'.format(name_of_file))