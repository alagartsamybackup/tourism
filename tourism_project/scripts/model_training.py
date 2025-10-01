"""
Model Training Script
Trains multiple models and uploads the best one to Hugging Face
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
from huggingface_hub import HfApi, create_repo
import joblib
import os

def main():
    # Load processed data
    X_train = pd.read_csv('tourism_project/data/X_train.csv')
    X_test = pd.read_csv('tourism_project/data/X_test.csv')
    y_train = pd.read_csv('tourism_project/data/y_train.csv').squeeze()
    y_test = pd.read_csv('tourism_project/data/y_test.csv').squeeze()
    
    # Define models
    models = {
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {'max_depth': [3, 5, 7]}
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {'n_estimators': [50, 100]}
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=42, eval_metric='logloss'),
            'params': {'n_estimators': [50, 100]}
        }
    }
    
    # Train models
    best_models = {}
    results = []
    
    mlflow.set_experiment("Tourism_Package_Prediction")
    
    for model_name, model_info in models.items():
        with mlflow.start_run(run_name=model_name):
            grid_search = GridSearchCV(
                model_info['model'],
                model_info['params'],
                cv=5,
                scoring='f1',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_models[model_name] = best_model
            
            # Evaluate
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Log to MLflow
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metrics({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc
            })
            mlflow.sklearn.log_model(best_model, model_name)
            
            results.append({
                'Model': model_name,
                'F1-Score': f1,
                'ROC-AUC': roc_auc,
                'Best Params': grid_search.best_params_
            })
    
    # Select best model
    results_df = pd.DataFrame(results)
    best_result = results_df.loc[results_df['F1-Score'].idxmax()]
    best_model_name = best_result['Model']
    best_model = best_models[best_model_name]
    
    # Save best model
    joblib.dump(best_model, 'tourism_project/model_building/best_model.pkl')
    
    # Upload to Hugging Face
    HF_TOKEN = os.getenv('HF_TOKEN')
    model_repo_name = "tourism-package-prediction-model"
    
    api = HfApi()
    
    # Upload model and preprocessing files
    model_files = [
        'tourism_project/model_building/best_model.pkl',
        'tourism_project/data/scaler.pkl',
        'tourism_project/data/label_encoders.pkl'
    ]
    
    for file_path in model_files:
        if os.path.exists(file_path):
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=os.path.basename(file_path),
                repo_id=f"alagarst/{model_repo_name}",
                repo_type="model",
                token=HF_TOKEN
            )
    
    print(f"Best model ({best_model_name}) trained and uploaded to Hugging Face!")
    print(f"Best F1-Score: {best_result['F1-Score']:.4f}")

if __name__ == "__main__":
    main()
