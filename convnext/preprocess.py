"""
    Author: Adrian Segura Lorente
    Description: This file serves the purpose of generating some important information
    related to the raw dataset to enable the establishment of a dataset
    and afterwards training.
"""

import os
import pandas as pd


"""
    This function generates data frame having an image path at index 1
    and an image label (e.g. which cow it belongs to) at index 2
"""
def generate_annotations(directory: str) -> None:
    dirnames = os.listdir(directory)
    dirnames.sort() # order by filename
    dirnames = dirnames[1:]
    indices = []
    labels = []
    for dname in dirnames:
        indices = indices + os.listdir(directory+'/'+dname+'/')
        labels = labels + [dname for lb in os.listdir(directory+'/'+dname+'/')]
    df = pd.DataFrame({'Path': indices,'Label': labels})
    df.to_csv('annotations.csv')