"""
Data Preparation Script
Cleans, processes, and splits the dataset
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from huggingface_hub import HfApi
import joblib
import os

def main():
    # Load data
    df = pd.read_csv('tourism_project/data/tourism.csv')

    # Data cleaning
    df_cleaned = df.drop(['CustomerID'], axis=1)

    # Handle missing values
    numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
    categorical_columns = df_cleaned.select_dtypes(include=['object']).columns

    imputer_numeric = SimpleImputer(strategy='median')
    df_cleaned[numeric_columns] = imputer_numeric.fit_transform(df_cleaned[numeric_columns])

    imputer_categorical = SimpleImputer(strategy='most_frequent')
    df_cleaned[categorical_columns] = imputer_categorical.fit_transform(df_cleaned[categorical_columns])

    # Encode categorical variables
    label_encoders = {}
    for column in categorical_columns:
        le = LabelEncoder()
        df_cleaned[column] = le.fit_transform(df_cleaned[column].astype(str))
        label_encoders[column] = le

    # Split data
    X = df_cleaned.drop('ProdTaken', axis=1)
    y = df_cleaned['ProdTaken']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save processed data
    pd.DataFrame(X_train_scaled, columns=X.columns).to_csv('tourism_project/data/X_train.csv', index=False)
    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv('tourism_project/data/X_test.csv', index=False)
    y_train.to_csv('tourism_project/data/y_train.csv', index=False)
    y_test.to_csv('tourism_project/data/y_test.csv', index=False)

    # Save preprocessing objects and training column names
    joblib.dump(label_encoders, 'tourism_project/data/label_encoders.pkl')
    joblib.dump(scaler, 'tourism_project/data/scaler.pkl')
    joblib.dump(imputer_numeric, 'tourism_project/data/imputer_numeric.pkl')
    joblib.dump(imputer_categorical, 'tourism_project/data/imputer_categorical.pkl')
    joblib.dump(X_train.columns.tolist(), 'tourism_project/data/X_train_columns.pkl')


    # Upload to Hugging Face
    HF_TOKEN = os.getenv('HF_TOKEN')
    repo_name = "tourism-package-prediction-dataset"

    api = HfApi()
    files_to_upload = [
        'tourism_project/data/X_train.csv',
        'tourism_project/data/X_test.csv',
        'tourism_project/data/y_train.csv',
        'tourism_project/data/y_test.csv',
        'tourism_project/data/scaler.pkl',
        'tourism_project/data/label_encoders.pkl',
        'tourism_project/data/imputer_numeric.pkl',
        'tourism_project/data/imputer_categorical.pkl',
        'tourism_project/data/X_train_columns.pkl' # Upload the new file
    ]

    for file_path in files_to_upload:
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=os.path.basename(file_path),
            repo_id=f"alagarst/{repo_name}",
            repo_type="dataset",
            token=HF_TOKEN
        )
        print(f"Uploaded {os.path.basename(file_path)} to {repo_name}")


    print("Data preparation completed and uploaded to Hugging Face!")

if __name__ == "__main__":
    main()
