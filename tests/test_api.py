import pytest
from fastapi.testclient import TestClient
from src.main import app  # ← use absolute import
from src.feature_names import FEATURE_NAMES  # dynamically imported feature list
import random
import json

client = TestClient(app)

# Create a randomized valid feature set
def generate_random_features():
    return {feature: round(random.uniform(0.0, 1.0), 3) for feature in FEATURE_NAMES}

def test_predict_knn():
    payload = {
        "features": generate_random_features(),
        "model_name": "knn"
    }

    # save the payload for future use as a .json file
    with open("test_payload_knn.json", "w") as f:
        json.dump(payload, f, indent=2)

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert "probabilities" in data
    assert isinstance(data["probabilities"], dict)

def test_predict_invalid_model():
    payload = {
        "features": generate_random_features(),
        "model_name": "unknown_model"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "detail" in response.json()

def test_projection_pca():
    response = client.get("/projection?method=pca")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert "x" in data[0] and "y" in data[0] and "label" in data[0]

def test_projection_tsne():
    response = client.get("/projection?method=tsne")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert "x" in data[0] and "y" in data[0] and "label" in data[0]

def test_projection_invalid_method():
    response = client.get("/projection?method=abc")
    assert response.status_code == 400

def test_metrics_knn():
    response = client.get("/metrics/knn")
    assert response.status_code == 200
    data = response.json()
    assert "accuracy" in data
    assert "confusion_matrix" in data
    assert "labels" in data

def test_metrics_invalid_model():
    response = client.get("/metrics/invalid_model")
    assert response.status_code == 400