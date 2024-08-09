from fastapi import HTTPException

from models.predict import PredictRequest
from services.prediction_services.services import PredictorService


def predict_app(request: PredictRequest):
    predictor_service = PredictorService()

    try:
        shop_id = request.shop_id
        item_id = request.item_id
        predictor_service.run()
        prediction = predictor_service.make_prediction(shop_id=shop_id, item_id=item_id)
        return {"shop_id": shop_id, "item_id": item_id, "predictions": prediction}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction failed")