import kaggle
import os

os.makedirs('data', exist_ok=True)
kaggle.api.authenticate()
kaggle.api.dataset_download_files(
    'mlg-ulb/creditcardfraud',
    path='data/',
    unzip=True
)
print("Dataset downloaded!")
