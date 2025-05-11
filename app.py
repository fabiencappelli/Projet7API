from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "deployed_model"
device    = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device).eval()

app = FastAPI()

class Payload(BaseModel):
    text: str

@app.post("/predict")
def predict(payload: Payload):
    encoding = tokenizer(
        payload.text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}
    with torch.no_grad():
        logits = model(**encoding).logits
        probs  = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
    return {
        "probabilities": probs.tolist(),
        "prediction":     int(probs.argmax())
    }

'''
uvicorn app:app --host 0.0.0.0 --port 8000

curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text":"My tweet"}'
'''