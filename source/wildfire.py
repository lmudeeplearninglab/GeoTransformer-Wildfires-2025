"""
Author: Tinbite Ermias
Date: 9/25/25
Description:
"""

import pandas as pd
import numpy as np
from IPython.core.interactiveshell import InteractiveShell
import random
InteractiveShell.ast_node_interactivity = "all"
import os
from datetime import datetime
import plotly.express as px
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt


def main():

  
    '''
    Loading Datasets:

    # One method: (non tfrecord file) 
    wildfire = load_data('numeric.csv', header=1, predict_col=0)
    X = wildfire.X; Xnames = wildfire.Xnames
    y = wildfire.y; yname = wildfire.yname
    n,d = X.shape  # n = number of examples, d =  number of features

    print(X.shape)
    
    ### Other Method: (for tfrecord file)
    data_folder = Path(r"C:/GradResearch_GeoTransformer_Wildfires_2025/GeoTransformer-Wildfires-2025/data")

    # Find all .tfrecord files in that folder
    tfrecord_files = list(data_folder.glob("*.tfrecord"))
    print(f"Found {len(tfrecord_files)} TFRecord files")

    # Create a dataset from all of them
    raw_dataset = tf.data.TFRecordDataset([str(f) for f in tfrecord_files])

    for raw_record in raw_dataset.take(2):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        print(example)

    Load all tfrecord files
    Path to your TFRecord file
    '''

if __name__ == '__main__':
    main()
