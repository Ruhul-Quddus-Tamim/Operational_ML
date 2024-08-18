import os

# Set environment variables for Kaggle API
KAGGLE_USERNAME = os.environ['KAGGLE_USERNAME']
KAGGLE_KEY = os.environ['KAGGLE_KEY']

# Import Kaggle API after setting environment variables
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset(dataset, file_name, download_path):
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Download the dataset
    api.dataset_download_file(dataset, file_name, path=download_path)

    # Check if the file has been downloaded
    file_path = os.path.join(download_path, file_name)
    if os.path.exists(file_path + ".zip"):
        print(f"Downloaded {file_name} successfully.")
        return True
    else:
        print(f"Failed to download {file_name}.")
        return False

if __name__ == "__main__":
    dataset = 'birdy654/eeg-brainwave-dataset-feeling-emotions'
    file_name = 'emotions.csv'
    download_path = 'EEG_Brainwave/data'
    success = download_dataset(dataset, file_name, download_path)
    
    if success:
        # Unzip the file if download was successful
        os.system(f'unzip {download_path}/{file_name}.zip -d {download_path}')
