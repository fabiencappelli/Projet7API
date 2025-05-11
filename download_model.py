''' download_model.py

Goal is to use just one time, in order to download files from MLFlow

I use if __name__ == "__main__" so the code of the loop just launches when we directly run the file
I do not plan yet to add other functions or classes

'''
import os
from dotenv import load_dotenv
import shutil

import mlflow.pytorch
import dagshub
import torch
from transformers import AutoTokenizer

# Secrets are in the .env file
# I might have to do differently once online
load_dotenv()
dagshub_token = os.getenv('DAGSHUB_TOKEN')

# Initialisation Dagshub
dagshub.auth.add_app_token(dagshub_token)
dagshub.init(repo_owner='fabiencappelli', repo_name='Projet_07', mlflow=True)

# Vars
MLFLOW_MODEL_URI = "runs:/bba69ea985b943e196e631cfb950505f/model"
LOCAL_MODEL_DIR  = "deployed_model"
CHECKPOINT       = "distilbert-base-uncased"
MAXLEN           = 128

if __name__ == "__main__":
    # 1) Reset du dossier
    if os.path.exists(LOCAL_MODEL_DIR):
        shutil.rmtree(LOCAL_MODEL_DIR)
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

    # 2) Charger le modèle MLflow
    hf_model = mlflow.pytorch.load_model(
        model_uri=MLFLOW_MODEL_URI,
        map_location=torch.device("cpu")
    )
    hf_model.eval()

    # 3) Sauvegarde HF-style à la racine
    hf_model.save_pretrained(LOCAL_MODEL_DIR)

    # 4) Sauvegarde du tokenizer à la racine
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    tokenizer.save_pretrained(LOCAL_MODEL_DIR)

    # 5) Supprimer l’ancien dossier MLflow
    old_mlflow_dir = os.path.join(LOCAL_MODEL_DIR, "model")
    if os.path.isdir(old_mlflow_dir):
        shutil.rmtree(old_mlflow_dir)
