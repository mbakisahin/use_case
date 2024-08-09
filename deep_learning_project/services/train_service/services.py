from utils.log.logger import get_logger
from services.train_service.training_pipeline.train import TrainingPipeline  # Use the actual path and module
from utils.networks.dlmodel import DPModel
from utils.networks.dlmodel.losses.mse import LossMSE
from utils.networks.dlmodel.optimizers.adam import OptimizerAdam

logger = get_logger(__name__)

class TrainingService:
    """
    Service for training a deep learning model.
    """

    def __init__(self, layer_architecture, batch_size, time_step, train_ratio, learning_rate, epochs, n_components):
        """
        Initialize the TrainingService with the given parameters.

        Args:
            layer_architecture (list): Architecture of the neural network layers.
            batch_size (int): Size of each training batch.
            time_step (int): Number of time steps for the model.
            train_ratio (float): Ratio of the data to be used for training.
            learning_rate (float): Learning rate for the optimizer.
            epochs (int): Number of training epochs.
            n_components (int): Number of PCA components.
        """
        self.layer_architecture = layer_architecture
        self.batch_size = batch_size
        self.time_step = time_step
        self.train_ratio = train_ratio
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.n_components = n_components

    def train_model(self):
        """
        Train the model using the specified training pipeline.
        """
        # Initialize the training pipeline
        pipeline = TrainingPipeline(
            file_path='LAST_FILE_NAME',
            layer_architecture=self.layer_architecture,
            current_model=DPModel(),
            current_loss=LossMSE(),
            current_optimizer=OptimizerAdam,
            service='SERVICE_ACCOUNT_FILE',
            model_path='MODEL_SAVED',
            model_name='MODEL_NAME',
            scaler_name='SCALER_NAME',
            pca_name='PCA_NAME',
            batch_size=self.batch_size,
            time_step=self.time_step,
            train_ratio=self.train_ratio,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            n_components=self.n_components,
        )

        logger.info("Training started")
        pipeline.run()
        logger.info("Training completed")
