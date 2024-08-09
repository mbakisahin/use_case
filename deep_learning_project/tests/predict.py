from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_predict_route():
    response = client.post("/prediction", json={
        "shop_id": 110,
        "item_id": 107172
    })
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_invalid_input():
    response = client.post("/prediction", json={
        "shop_id": "invalid",
        "item_id": "invalid"
    })
    assert response.status_code == 422
    assert "detail" in response.json()


def test_missing_field():
    response = client.post("/prediction", json={
        "shop_id": 110
    })
    assert response.status_code == 422
    assert "detail" in response.json()

def test_invalid_shop_id():
    response = client.post("/prediction", json={
        "shop_id": 98,
        "item_id": 107172
    })
    assert response.status_code == 400
    assert "detail" in response.json()
    assert response.json()["detail"] == "Invalid shop_id. Must be between 99 and 158."

def test_invalid_item_id():
    response = client.post("/prediction", json={
        "shop_id": 110,
        "item_id": 99999
    })
    assert response.status_code == 400
    assert "detail" in response.json()
    assert response.json()["detail"] == "Invalid item_id. Must be between 100000 and 122169."

def test_invalid_both_ids():
    response = client.post("/prediction", json={
        "shop_id": 98,
        "item_id": 99999
    })
    assert response.status_code == 400
    assert "detail" in response.json()






