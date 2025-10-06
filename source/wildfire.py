"""
Author: Tinbite Ermias
Date: 9/25/25
Description:
"""

# Use only the provided packages!
import math
import csv
from util import *

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


def main():
    # load dataset
    wildfire = load_data('b8aeb030-140d-43d2-aa29-1a80862e3d62.csv', header=1, predict_col=0)
    X = wildfire.X; Xnames = wildfire.Xnames
    y = wildfire.y; yname = wildfire.yname
    n,d = X.shape  # n = number of examples, d =  number of features