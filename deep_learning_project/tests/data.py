from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_data_route():
    response = client.post("/data")
    assert response.status_code == 200
    assert response.json() == {"key": "value"}
