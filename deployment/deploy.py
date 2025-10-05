#!/usr/bin/env python3
from huggingface_hub import HfApi, create_repo
import os

def deploy_to_huggingface():
    api = HfApi()
    space_name = 'alagarst/tourism-package-prediction'
    
    try:
        create_repo(repo_id=space_name, repo_type='space', space_sdk='streamlit', exist_ok=True)
        print(f'Space created: {space_name}')
        
        files = [('app.py', 'app.py'), ('requirements.txt', 'requirements.txt'), ('Dockerfile', 'Dockerfile')]
        
        for local, remote in files:
            api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=space_name, repo_type='space')
            print(f'Uploaded {remote}')
        
        print(f'App URL: https://huggingface.co/spaces/{space_name}')
        
    except Exception as e:
        print(f'Error: {e}')
        return False
    
    return True

if __name__ == '__main__':
    deploy_to_huggingface()
