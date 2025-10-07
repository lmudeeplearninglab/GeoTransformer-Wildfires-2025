import os
import zipfile
import kaggle
import pandas as pd
import numpy as np
from IPython.core.interactiveshell import InteractiveShell
import random
InteractiveShell.ast_node_interactivity = "all"
from datetime import datetime
import plotly.express as px
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

def main():
    all_csv_files = glob.glob('../data/**/*.csv', recursive=True)
    print(len(all_csv_files))


if __name__ == '__main__':
    main()























'''
IF USING KAGGLE:

# Define the dataset identifier and download directory
dataset_id = 'fantineh/next-day-wildfire-spread'
download_dir = 'C:/GradResearch_GeoTransformer_Wildfires_2025/GeoTransformer-Wildfires-2025/data'
zip_file_path = os.path.join(download_dir, 'next-day-wildfire-spread.zip')

# Create download directory if it doesn't exist
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

# Download the dataset
kaggle.api.dataset_download_files(dataset_id, path=download_dir, unzip=False)

# Check if the zip file already exists, remove the old file if it does
if os.path.exists(zip_file_path):
    os.remove(zip_file_path)

# Download the latest dataset again
kaggle.api.dataset_download_files(dataset_id, path=download_dir, unzip=False)

# Extract the dataset
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(download_dir)

print(f'Dataset updated and extracted to {download_dir}')

Kaggle Data Sets:
> 'fantineh/next-day-wildfire-spread'
> 'avkashchauhan/california-wildfire-dataset-from-2000-2021'

'''
