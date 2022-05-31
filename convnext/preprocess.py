"""
    Author: Adrian Segura Lorente
    Description: This file serves the purpose of generating some important information
    related to the raw dataset to enable the establishment of a dataset
    and afterwards training.
"""

import os
from numpy import number
import pandas as pd

def generate_folds(directory:str, k: int):
    try: # case directory does not exist
        os.mkdir('./training_testing_folds')
    except FileExistsError:
        pass # else do nothing and just overwrite files

    files = os.listdir(directory)
    files.sort()
    files = files[1:] # first element is never a valid file but a .ds file, so skip it
    # print(files)
    labels = [file.split('_')[0] for file in files]
    
    number_files = len(files)-1

    validation_size = int(number_files/k)
    train_size = number_files-validation_size

    # process:
    # generate list of all files and from those, use a loop to generate each fold and generate csv files
    for i in range(0,k):
        train_current = []
        validation_current = []
        upper_limit_validation = validation_size*i+validation_size if validation_size*i+validation_size <= number_files else number_files
        
        validation_current = files[validation_size*i:upper_limit_validation] # select only the specific portion for the validation at fold k
        labels_validation_current = [file.split('_')[0] for file in validation_current] # generate labels for k'th validation fold
        df = pd.DataFrame({'Path': validation_current,'Label': labels_validation_current}) # generate data frame having as columns the path to an image and labels for each of such images
        df.to_csv('./training_testing_folds/validation_annotations_fold{}.csv'.format(i))


        training_current = files[0:validation_size*i]+files[upper_limit_validation:] # select only the portion for the training at fold k
        labels_training_current = [file.split('_')[0] for file in training_current] # generate labels for k'th training fold
        df = pd.DataFrame({'Path': training_current,'Label': labels_training_current}) # generate data frame having as columns the path to an image and labels for each of such images
        df.to_csv('./training_testing_folds/training_annotations_fold{}.csv'.format(i))

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