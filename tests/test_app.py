import pytest
from app import app

'''
TestClient est une classe fournie par FastAPI (via son socle Starlette) qui vous permet de simuler des requêtes HTTP en Python, sans lancer un vrai serveur web.
'''
from fastapi.testclient import TestClient

client = TestClient(app)

def test_prediction_consistency():
    """
    Vérifie que prediction == argmax(probas).
    """
    r = client.post("/predict", json={"text": "Consistance"})
    data = r.json()
    probas = data["probabilities"]
    # argmax correspond au prediction
    assert data["prediction"] == probas.index(max(probas))

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

def test_schema():
    r = client.post("/predict", json={"text":"Bonjour"})
    data = r.json()
    # schéma
    assert "probabilities" in data and "prediction" in data
    assert isinstance(data["probabilities"], list) and len(data["probabilities"])==2
    assert isinstance(data["prediction"], int)