from utils.networks.dlmodel import Sigmoid, DPModel, LossMSE
from utils.networks.dlmodel.optimizers import OptimizerAdam
from services.train_service.training_pipeline.setup.train_pipeline import TrainingPipeline

if __name__ == '__main__':
    """
    Main script for running the training pipeline and saving the trained model, scaler, and PCA.

    This script loads the filtered data, defines the target column and features, specifies the layer architecture,
    initializes and runs the training pipeline, and saves the trained model, scaler, and PCA.
    """

    # Define the layer architecture
    layer_architecture = [
        {'output_size': 64, 'activation': Sigmoid},
        {'output_size': 64, 'activation': Sigmoid},
        {'output_size': 1, 'activation': None}
    ]

    # Initialize the training pipeline
    pipeline = TrainingPipeline(
        file_path='LAST_FILE_NAME',
        layer_architecture=layer_architecture,
        current_model=DPModel(),
        current_loss=LossMSE(),
        current_optimizer=OptimizerAdam,
        service='SERVICE_ACCOUNT_FILE',
        model_path='MODEL_SAVED',
        model_name='MODEL_NAME',
        scaler_name='SCALER_NAME',
        pca_name='PCA_NAME',
        batch_size=8192,
        time_step=10,
        train_ratio=0.8,
        learning_rate=0.1,
        epochs=100,
        n_components=10,
    )

    # Run the training pipeline
    pipeline.run()

