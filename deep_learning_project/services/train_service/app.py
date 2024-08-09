from fastapi import HTTPException

from utils.networks.dlmodel import Sigmoid, ReLU, Tanh
from models.train import TrainRequest
from services.train_service.services import TrainingService

def train_app(request: TrainRequest):
    """
    Train a model based on the given training request.

    Args:
        request (TrainRequest): Request object containing training parameters.

    Returns:
        dict: Status message indicating training completion.

    Raises:
        HTTPException: If training fails.
    """
    try:
        # Activation functions dictionary
        activation_functions = {
            "Sigmoid": Sigmoid,
            "Relu": ReLU,
            "Tanh": Tanh,
            None: None
        }

        # Convert string activations to actual functions
        layer_architecture = [
            {
                "output_size": layer.output_size,
                "activation": activation_functions[layer.activation]
            }
            for layer in request.layer_architecture
        ]

        training_service = TrainingService(
            layer_architecture=layer_architecture,
            batch_size=request.batch_size,
            time_step=request.time_step,
            train_ratio=request.train_ratio,
            learning_rate=request.learning_rate,
            epochs=request.epochs,
            n_components=request.n_components
        )
        training_service.train_model()
        return {"status": "training completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
