from fastapi import APIRouter, HTTPException

from models.train import TrainRequest
from services.train_service.app import train_app

router = APIRouter()


@router.post("/train")
def train(request: TrainRequest):
    """
    Trigger the training process.

    Args:
        request (TrainRequest): Training parameters.

    Returns:
        dict: Confirmation message.
    """
    # Parametre doğrulama
    if request.batch_size >= 10000:
        raise HTTPException(status_code=400, detail="Batch size too large. Maximum allowed is 9999.")
    if request.time_step >= 100:
        raise HTTPException(status_code=400, detail="Time step too large. Maximum allowed is 99.")
    if request.learning_rate >= 1.0:
        raise HTTPException(status_code=400, detail="Learning rate too large. Maximum allowed is 0.99.")
    if request.epochs >= 1000:
        raise HTTPException(status_code=400, detail="Too many epochs. Maximum allowed is 999.")
    if request.n_components >= 50:
        raise HTTPException(status_code=400, detail="Number of components too large. Maximum allowed is 49.")

    # Yeni eklenen parametre doğrulama
    if request.batch_size <= 0:
        raise HTTPException(status_code=400, detail="Batch size too small. Minimum allowed is 1.")
    if request.time_step <= 0:
        raise HTTPException(status_code=400, detail="Time step too small. Minimum allowed is 1.")
    if request.learning_rate <= 0:
        raise HTTPException(status_code=400, detail="Learning rate too small. Minimum allowed is 0.001.")
    if request.epochs <= 0:
        raise HTTPException(status_code=400, detail="Too few epochs. Minimum allowed is 1.")
    if request.n_components <= 0:
        raise HTTPException(status_code=400, detail="Number of components too small. Minimum allowed is 1.")

    # Katman mimarisi doğrulaması
    for layer in request.layer_architecture:
        if layer.output_size > 100:
            raise HTTPException(status_code=400, detail="Layer output size too large. Maximum allowed is 100.")
        if layer.output_size <= 0:
            raise HTTPException(status_code=400, detail="Layer output size too small. Minimum allowed is 1.")

    train_app(request)
    return {"message": "Training started"}
