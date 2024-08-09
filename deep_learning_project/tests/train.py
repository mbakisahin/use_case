from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_train_route():
    response = client.post("/train", json={
        "layer_architecture": [
            {"output_size": 64, "activation": "Sigmoid"},
            {"output_size": 32, "activation": "Sigmoid"},
            {"output_size": 1, "activation": None}
        ],
        "batch_size": 8192,
        "time_step": 10,
        "train_ratio": 0.8,
        "learning_rate": 0.01,
        "epochs": 10,
        "n_components": 10
    })
    assert response.status_code == 200
    assert response.json() == {"message": "Training started"}

def test_train_route_missing_params():
    response = client.post("/train", json={
        "layer_architecture": [
            {"output_size": 64, "activation": "Sigmoid"}
        ],
        "batch_size": 8192,
        "time_step": 10,
        "train_ratio": 0.8,
        "learning_rate": 0.01
    })
    assert response.status_code == 422

def test_train_route_invalid_params():
    response = client.post("/train", json={
        "layer_architecture": [
            {"output_size": "sixty-four", "activation": "Sigmoid"},
            {"output_size": 32, "activation": "Sigmoid"},
            {"output_size": 1, "activation": None}
        ],
        "batch_size": "eight-thousand",
        "time_step": 10,
        "train_ratio": 0.8,
        "learning_rate": "0.01",
        "epochs": 10,
        "n_components": 10
    })
    assert response.status_code == 422

def test_train_route_empty_body():
    response = client.post("/train", json={})
    assert response.status_code == 422

def test_train_route_no_body():
    response = client.post("/train")
    assert response.status_code == 422

def test_train_route_maximum_values():
    response = client.post("/train", json={
        "layer_architecture": [
            {"output_size": 1000, "activation": "Sigmoid"},
            {"output_size": 500, "activation": "Sigmoid"},
            {"output_size": 1, "activation": None}
        ],
        "batch_size": 10000,
        "time_step": 100,
        "train_ratio": 1.0,
        "learning_rate": 1.0,
        "epochs": 1000,
        "n_components": 100
    })
    assert response.status_code == 400
    assert "detail" in response.json()

def test_train_route_minimum_values():
    response = client.post("/train", json={
        "layer_architecture": [
            {"output_size": 1, "activation": "Sigmoid"},
            {"output_size": 1, "activation": "Sigmoid"},
            {"output_size": 1, "activation": None}
        ],
        "batch_size": 1,
        "time_step": 1,
        "train_ratio": 0.01,
        "learning_rate": 0.0001,
        "epochs": 1,
        "n_components": 1
    })
    assert response.status_code == 200
    assert response.json() == {"message": "Training started"}

def test_train_route_n_components_too_large():
    response = client.post("/train", json={
        "layer_architecture": [
            {"output_size": 1, "activation": "Sigmoid"},
            {"output_size": 1, "activation": "Sigmoid"},
            {"output_size": 1, "activation": None}
        ],
        "batch_size": 1,
        "time_step": 1,
        "train_ratio": 0.01,
        "learning_rate": 0.0001,
        "epochs": 1,
        "n_components": 100
    })
    assert response.status_code == 400
    assert "detail" in response.json()

def test_train_route_valid_large_values():
    response = client.post("/train", json={
        "layer_architecture": [
            {"output_size": 500, "activation": "Sigmoid"},
            {"output_size": 250, "activation": "Sigmoid"},
            {"output_size": 1, "activation": None}
        ],
        "batch_size": 5000,
        "time_step": 50,
        "train_ratio": 0.9,
        "learning_rate": 0.5,
        "epochs": 50,
        "n_components": 10
    })
    assert response.status_code == 400
    assert response.json()["detail"] == "Layer output size too large. Maximum allowed is 100."
