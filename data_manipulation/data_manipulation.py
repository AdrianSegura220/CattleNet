"""
    Author: Adrian Segura Lorente
    Description: This file serves the purpose of generating some important information
    related to the raw dataset to enable the establishment of a dataset
    and afterwards training.
"""

import os
import pandas as pd
import itertools
import math
import random

"""
    This function generates data frame having an image path at index 1
    and an image label (e.g. which cow it belongs to) at index 2
"""
def generate_annotations(directory: str,percentage=0.2) -> None:
    dirnames = os.listdir(directory)
    dirnames.sort() # order by filename
    dirnames = dirnames[1:]
    train_paths = []
    train_labels = []
    test_paths = []
    test_labels = []
    for dname in dirnames:
        dir_images = os.listdir(directory+'/'+dname+'/')
        len_directory = len(dir_images) # number of images for directory belonging to a cow
        for_test = 0
        # if there are more than two images of a cow, at least one will be in the test data set
        if len_directory > 2:
            for_test = math.ceil(percentage*len_directory) if len_directory < 10 else int(percentage*len_directory) # define how many images ough to be selected for the test set

            train_paths = train_paths + dir_images[:(len_directory-for_test)] # select every image, except for the ones used for testing
            train_labels = train_labels + [dname for lb in dir_images[:(len_directory-for_test)]] # set labels for images selected for training

            test_paths = test_paths + dir_images[(len_directory-for_test):] # select remaining portion (destined for testing set)
            test_labels = test_labels + [dname for lb in dir_images[(len_directory-for_test):]]
        else: # if there are less than 3 images, they will be used for training
            train_paths = train_paths + os.listdir(directory+'/'+dname+'/') # list all file names inside directory dname (where dname is the label of each cow. e.g. 0089)
            train_labels = train_labels + [dname for lb in os.listdir(directory+'/'+dname+'/')] # for each image in each directory append label
    # df = pd.DataFrame({'Path': indices,'Label': labels}) # generate data frame having as columns the path to an image and labels for each of such images
    return train_paths,train_labels,test_paths,test_labels
    # df.to_csv('annotations.csv')


def generate_balanced_annotations(directory: str) -> None:
    dirnames = os.listdir(directory)
    dirnames.sort() # order by filename
    dirnames = dirnames[1:]
    indices = []
    labels = []
    for dname in dirnames:
        indices = indices + os.listdir(directory+'/'+dname+'/') # list all file names inside directory dname (where dname is the label of each cow. e.g. 0089)
        labels = labels + [dname for lb in os.listdir(directory+'/'+dname+'/')] # for each image in each directory append label
    # df = pd.DataFrame({'Path': indices,'Label': labels}) # generate data frame having as columns the path to an image and labels for each of such images
    return indices,labels
    # df.to_csv('annotations.csv')


"""
    subset_combinations is a percentage (e.g. 0.1 for 10%)
"""
def generate_combinations(directory: str,percentage,subset_combinations):
    train_paths,train_labels,test_paths,test_labels = generate_annotations(directory,percentage)

    zipped_training = list(zip(train_paths,train_labels)) # zip paths and labels. e.g. given [path1.jpg,path2.jpg] and labels [1,2] => [(path1.jpg,1),(path2.jpg,2)]
    
    combinations_training = list(itertools.combinations(zipped_training,2)) # generate combination pairs, e.g. [((path1.jpg,1),(path2,2))]

    zipped_testing = list(zip(test_paths,test_labels))

    combinations_testing = list(itertools.combinations(zipped_testing,2))
    
    #shuffle combinations:
    random.shuffle(combinations_training)
    random.shuffle(combinations_testing)

    # only use subset of combinations:
    combinations_training = combinations_training[:int(len(combinations_training)*subset_combinations)]
    combinations_testing = combinations_testing[:int(len(combinations_testing)*subset_combinations)]


    # unzip combinations to have separate arrays in the form: [tuple1] [tuple2]
    # this is done because the itertools combinations produces tuples which we need to unzip in order to cleanly generate the data frame
    unzipped_training_combinations = zip(*combinations_training)
    unzipped_testing_combinations = zip(*combinations_testing)

    # unzipping produces two big tuples containing the left part of the tuple combinations and the right part of the tuples combinations
    # , so we have to convert these to lists. E.g. if we have [((1.jpg,1),(2.jpg,2),(3.jpg,3)),((1combination.jpg,someimagelabel),(2combination.jpg,someimagelabel),(3combination.jpg,someimagelabel))]
    # then we would get [[(1.jpg,1),(2.jpg,2),(3.jpg,3)],[(1combination.jpg,someimagelabel),(2combination.jpg,someimagelabel),(3combination.jpg,someimagelabel)]]:
    unzipped_training_combinations = [list(elem) for elem in unzipped_training_combinations]
    unzipped_testing_combinations = [list(elem) for elem in unzipped_testing_combinations]
    # the result is of the form: [[tuple1,...,tupleN],[tuple1_matchingPair,...,tupleN_matchingPair]]

    left_tuples_training = unzipped_training_combinations[0] # left part of combinations (i.e. all first images of pairs of images produced in the combination process)
    right_tuples_training = unzipped_training_combinations[1] # right part of combinations

    left_tuples_testing = unzipped_testing_combinations[0] # left part of combinations (i.e. all first images of pairs of images produced in the combination process)
    right_tuples_testing = unzipped_testing_combinations[1] # right part of combinations

    # In order to have our data frame accessible with indices as 'Path1', 'Label1', 'Path2', 'Label2' we must unzip the lists again, since they contain tuples
    # and transform them to lists
    unzipped_left_tuples_training = zip(*left_tuples_training)
    unzipped_right_tuples_training =  zip(*right_tuples_training)

    unzipped_left_tuples_testing = zip(*left_tuples_testing)
    unzipped_right_tuples_testing = zip(*right_tuples_testing)

    # the result is a list of tuples of the form [('path1,jpg','path2.jpg',...,'pathN.jpg'),('1','2',...,'N')]
    # and we need: paths1 = ['path1,jpg','path2.jpg',...,'pathN.jpg'] and labels1 = ['1','2',...,'N'], same for right side of tuples

    left_tuples_training_lists = [list(elem) for elem in unzipped_left_tuples_training]
    right_tuples_training_lists = [list(elem) for elem in unzipped_right_tuples_training]

    left_tuples_testing_lists = [list(elem) for elem in unzipped_left_tuples_testing]
    right_tuples_testing_lists = [list(elem) for elem in unzipped_right_tuples_testing]

    # paths from first image in each pair from combinations
    left_tuples_paths_training = left_tuples_training_lists[0]
    # labels from first image in each pair from combinations
    left_tuples_labels_training = left_tuples_training_lists[1]
    # paths from second image in each pair from combinations
    right_tuples_paths_training = right_tuples_training_lists[0]
    # labels from second image in each pair from combinations (e.g. from pair: (img1.jpg,'label 1'),(img2.jpg,'label 2') ; then 'label 2' would be part of this list)
    right_tuples_labels_training = right_tuples_training_lists[1]


    # paths from first image in each pair from combinations
    left_tuples_paths_testing= left_tuples_testing_lists[0]
    # labels from first image in each pair from combinations
    left_tuples_labels_testing= left_tuples_testing_lists[1]
    # paths from second image in each pair from combinations
    right_tuples_paths_testing= right_tuples_testing_lists[0]
    # labels from second image in each pair from combinations (e.g. from pair: (img1.jpg,'label 1'),(img2.jpg,'label 2') ; then 'label 2' would be part of this list)
    right_tuples_labels_testing= right_tuples_testing_lists[1]


    # finally, define training data frame with lists containing all paths and labels for pairs of images generated from the combinations 
    training_df = pd.DataFrame({'Path1': left_tuples_paths_training,'Label1': left_tuples_labels_training,'Path2':right_tuples_paths_training,'Label2':right_tuples_labels_training})

    # also, define testing data frame with lists containing all paths and labels for pairs of images generated from the combinations 
    testing_df = pd.DataFrame({'Path1': left_tuples_paths_testing,'Label1': left_tuples_labels_testing,'Path2':right_tuples_paths_testing,'Label2':right_tuples_labels_testing})

    # generate csv files for both data frames, so they can be used with the data loader in models
    training_df.to_csv('annotations_training.csv')
    testing_df.to_csv('annotations_testing.csv')