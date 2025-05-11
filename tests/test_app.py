import os
import torch
from unittest.mock import patch
import pytest

class DummyTokenizer:
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, text, max_length, padding, truncation, return_tensors):
        # renvoie exactement ce que FastAPI attend
        return {
            "input_ids":      torch.zeros((1, max_length), dtype=torch.long),
            "attention_mask": torch.ones((1, max_length), dtype=torch.long),
        }

class DummyModel:
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, **inputs):
        # renvoie toujours des logits constants [0.2, 0.8]
        class Out:
            logits = torch.tensor([[0.2, 0.8]])
        return Out()
    def to(self, device):
        return self
    def eval(self):
        return self

# 2) Avant d'importer app, on patch les méthodes de chargement HF
with patch("transformers.AutoTokenizer.from_pretrained", return_value=DummyTokenizer()), \
     patch("transformers.AutoModelForSequenceClassification.from_pretrained", return_value=DummyModel()):
    # On importe la FastAPI app *après* avoir patché
    from fastapi.testclient import TestClient
    from app import app

client = TestClient(app)

def test_status_codes_ok_and_empty():
    # bon payload
    r1 = client.post("/predict", json={"text":"Bonjour"})
    assert r1.status_code == 200
    # texte vide
    r2 = client.post("/predict", json={"text":""})
    assert r2.status_code == 200
    '''
    200 status is success
    '''

def test_status_code_invalid():
    # pas de clé "text"
    r = client.post("/predict", json={})
    assert r.status_code == 422
    '''
    HTTP status code 422, known as "Unprocessable Entity," indicates that the server understands the content type of the request entity and the syntax of the request entity is correct, but it was unable to process the contained instructions
    '''

def test_prediction_consistency():
    """
    Vérifie que prediction == argmax(probas).
    """
    r = client.post("/predict", json={"text": "Consistance"})
    data = r.json()
    probas = data["probabilities"]
    # argmax correspond au prediction
    assert data["prediction"] == probas.index(max(probas))


def test_schema():
    r = client.post("/predict", json={"text":"Bonjour"})
    data = r.json()
    # schéma
    assert "probabilities" in data and "prediction" in data
    assert isinstance(data["probabilities"], list) and len(data["probabilities"])==2
    assert isinstance(data["prediction"], int)
