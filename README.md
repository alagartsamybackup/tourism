# Tourism Wellness Package Prediction - MLOps Project

## Project Overview

MLOps pipeline for predicting customer likelihood of purchasing wellness tourism packages.

## Architecture

```
tourism_project/
├── data/
├── model_building/
├── deployment/
├── scripts/
└── .github/workflows/
```

## Features

- Data versioning on Hugging Face
- Model training with MLflow
- Streamlit web application
- CI/CD with GitHub Actions

## Links

- Web App: https://huggingface.co/spaces/alagarst/tourism-package-prediction
- Dataset: https://huggingface.co/datasets/alagarst/tourism-wellness-dataset
- Model: https://huggingface.co/alagarst/tourism-wellness-model
- GitHub: https://github.com/alagartsamybackup/tourism

## Performance

- Best Model: GradientBoosting
- Accuracy: 0.9395

## Setup

1. Clone repository
2. Install dependencies
3. Add HF_TOKEN to GitHub Secrets
4. Push to trigger pipeline
