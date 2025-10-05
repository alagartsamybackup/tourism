#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from huggingface_hub import hf_hub_download, HfApi
import pickle
import os

def prepare_data():
    api = HfApi(token=os.environ.get('HF_TOKEN'))
    repo_name = "alagarst/tourism-wellness-dataset"
    
    data_path = hf_hub_download(
        repo_id=repo_name,
        filename="tourism_raw.csv",
        repo_type="dataset"
    )
    
    df = pd.read_csv(data_path)
    
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)
    
    numerical_cols = ['Age', 'DurationOfPitch', 'NumberOfFollowups', 'NumberOfTrips', 'MonthlyIncome']
    for col in numerical_cols:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)
    
    categorical_cols = ['TypeofContact', 'Occupation', 'Gender', 'MaritalStatus', 'ProductPitched', 'Designation']
    label_encoders = {}
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    X = df.drop(['CustomerID', 'ProdTaken'], axis=1, errors='ignore')
    y = df['ProdTaken']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    train_data = pd.concat([
        pd.DataFrame(X_train_scaled, columns=X_train.columns),
        y_train.reset_index(drop=True)
    ], axis=1)
    
    test_data = pd.concat([
        pd.DataFrame(X_test_scaled, columns=X_test.columns),
        y_test.reset_index(drop=True)
    ], axis=1)
    
    train_data.to_csv('/tmp/train_data.csv', index=False)
    test_data.to_csv('/tmp/test_data.csv', index=False)
    
    api.upload_file(
        path_or_fileobj="/tmp/train_data.csv",
        path_in_repo="train_data.csv",
        repo_id=repo_name,
        repo_type="dataset"
    )
    
    api.upload_file(
        path_or_fileobj="/tmp/test_data.csv",
        path_in_repo="test_data.csv",
        repo_id=repo_name,
        repo_type="dataset"
    )
    
    print("Data preparation complete")

if __name__ == "__main__":
    prepare_data()
