import os
import zipfile
import kaggle

# Define the dataset identifier and download directory
dataset_id = 'vijayveersingh/the-california-wildfire-data'
download_dir = 'C:\GradResearch_GeoTransformer_Wildfires_2025\GeoTransformer-Wildfires-2025/data'
zip_file_path = os.path.join(download_dir, 'the-california-wildfire-data.zip')

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
