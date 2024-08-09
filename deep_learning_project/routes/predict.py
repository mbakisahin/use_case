from fastapi import APIRouter, HTTPException

from models.predict import PredictRequest
from services.prediction_services.app import predict_app

router = APIRouter()

@router.post("/prediction")
def prediction(request: PredictRequest):
    """
    Trigger the prediction process.

    Args:
        request (PredictRequest): Prediction parameters.

    Returns:
        None
    """
    # Validate presence of shop_id and item_id
    if request.shop_id is None and request.item_id is None:
        raise HTTPException(status_code=400, detail="Both shop_id and item_id are missing.")
    if request.shop_id is None:
        raise HTTPException(status_code=400, detail="shop_id is missing.")
    if request.item_id is None:
        raise HTTPException(status_code=400, detail="item_id is missing.")

    # Validate shop_id
    if not (99 <= request.shop_id <= 158):
        raise HTTPException(status_code=400, detail="Invalid shop_id. Must be between 99 and 158.")

    # Validate item_id
    if not (100000 <= request.item_id <= 122169):
        raise HTTPException(status_code=400, detail="Invalid item_id. Must be between 100000 and 122169.")

    prediction = predict_app(request)
    if prediction is None:
        return {"prediction": None}
    return {"prediction": prediction}
