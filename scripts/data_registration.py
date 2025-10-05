#!/usr/bin/env python3

import pandas as pd
from huggingface_hub import HfApi, create_repo
import os

def register_dataset():
    api = HfApi(token=os.environ.get('HF_TOKEN'))
    
    df = pd.read_csv('data/tourism.csv')
    print(f"Loaded dataset: {df.shape}")
    
    repo_name = "alagarst/tourism-wellness-dataset"
    
    create_repo(repo_id=repo_name, repo_type="dataset", exist_ok=True)
    
    api.upload_file(
        path_or_fileobj="data/tourism.csv",
        path_in_repo="tourism_raw.csv",
        repo_id=repo_name,
        repo_type="dataset"
    )
    
    print(f"Dataset uploaded to {repo_name}")

if __name__ == "__main__":
    register_dataset()
