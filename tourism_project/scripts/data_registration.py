"""
Data Registration Script
Uploads raw dataset to Hugging Face Hub
"""
import pandas as pd
from huggingface_hub import HfApi, create_repo
import os

def main():
    # Load dataset
    df = pd.read_csv('tourism_project/data/tourism.csv')

    # Save locally
    df.to_csv('tourism_project/data/tourism_raw.csv', index=False)

    # Upload to Hugging Face
    HF_TOKEN = os.getenv('HF_TOKEN')
    repo_name = "tourism-package-prediction-dataset"

    api = HfApi()
    api.upload_file(
        path_or_fileobj="tourism_project/data/tourism_raw.csv",
        path_in_repo="tourism_raw.csv",
        repo_id=f"alagarst/{repo_name}",
        repo_type="dataset",
        token=HF_TOKEN
    )
    print("Dataset uploaded to Hugging Face Hub successfully!")

if __name__ == "__main__":
    main()
