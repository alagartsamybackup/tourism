#!/usr/bin/env python3

import pandas as pd
import pickle
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
from huggingface_hub import hf_hub_download, HfApi, create_repo
import os

def train_models():
    api = HfApi(token=os.environ.get('HF_TOKEN'))
    dataset_repo = "alagarst/tourism-wellness-dataset"
    model_repo = "alagarst/tourism-wellness-model"
    
    train_path = hf_hub_download(repo_id=dataset_repo, filename="train_data.csv", repo_type="dataset")
    test_path = hf_hub_download(repo_id=dataset_repo, filename="test_data.csv", repo_type="dataset")
    
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    X_train = train_data.drop('ProdTaken', axis=1)
    y_train = train_data['ProdTaken']
    X_test = test_data.drop('ProdTaken', axis=1)
    y_test = test_data['ProdTaken']
    
    mlflow.set_experiment("Tourism_CI_CD")
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False)
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(model, name)
            
            print(f"{name}: {accuracy:.4f}")
            
            if accuracy > best_score:
                best_score = accuracy
                best_model = model
                best_name = name
    
    with open('/tmp/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    create_repo(repo_id=model_repo, repo_type="model", exist_ok=True)
    
    api.upload_file(
        path_or_fileobj="/tmp/best_model.pkl",
        path_in_repo="best_model.pkl",
        repo_id=model_repo,
        repo_type="model"
    )
    
    print(f"Best model ({best_name}) uploaded with accuracy: {best_score:.4f}")

if __name__ == "__main__":
    train_models()
