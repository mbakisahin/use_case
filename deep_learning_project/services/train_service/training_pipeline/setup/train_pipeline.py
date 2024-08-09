import os
from io import BytesIO

import joblib
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from utils.db import GoogleDriveHandler
from services.train_service.training_pipeline.data import ModelTrainer, DatasetProcessor
from utils.log.logger import get_logger

logger = get_logger(__name__)

class TrainingPipeline():
    """
    A class for building and running a training pipeline for a machine learning model.

    This class loads data, scales it, applies PCA, prepares datasets for training and testing,
    trains a model, and saves the model along with the scaler and PCA.
    """

    def __init__(self, file_path, layer_architecture,
                 current_model, current_loss, current_optimizer,
                 batch_size, service, model_path, model_name, scaler_name,
                 pca_name, time_step=10, train_ratio=0.8,
                 learning_rate=0.01, epochs=100, n_components=30):
        """
        Initializes the TrainingPipeline with specified parameters.

        Parameters:
        file_path (str): The path to the CSV data file.
        features (list): The list of feature column names.
        target (str): The target column name.
        layer_architecture (list): The architecture of the layers for the model.
        time_step (int, optional): The number of time steps to use for creating the dataset. Default is 10.
        train_ratio (float, optional): The ratio of the dataset to use for training. Default is 0.8.
        learning_rate (float, optional): The learning rate for the optimizer. Default is 0.01.
        epochs (int, optional): The number of epochs to train the model. Default is 100.
        n_components (int, optional): The number of principal components for PCA. Default is 30.

        Returns:
        None.
        """
        self.google_drive_handler = GoogleDriveHandler(service)

        self.file_path = os.getenv(file_path)
        self.features = None
        self.target = None
        self.layer_architecture = layer_architecture
        self.time_step = time_step
        self.train_ratio = train_ratio
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.n_components = n_components
        self.batch_size = batch_size
        self.current_model = current_model
        self.current_loss = current_loss
        self.current_optimizer = current_optimizer
        self.model_path = os.getenv(model_path)
        self.model_name = os.getenv(model_name)
        self.scaler_name = os.getenv(scaler_name)
        self.pca_name = os.getenv(pca_name)
        self.data = None
        self.scaled_data = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.model = None
        self.scaler = MinMaxScaler()
        self.pca = PCA(n_components=self.n_components)

    def load_data(self):
        """
        Loads the data from the CSV file.

        Returns:
        pandas.DataFrame: The loaded data.
        """

        file_id = self.google_drive_handler.search_file_by_name(self.file_path)

        if file_id:
            # Dosyayı indir
            downloaded_file = self.google_drive_handler.download_file_from_drive(file_id)

            self.data = pd.read_csv(downloaded_file)
            return self.data

    def create_features(self):
        # Specify the target column
        self.target = 'amount'
        self.features = [col for col in self.data.columns if col != self.target]
        self.features.remove('date')

    def scale_data(self):
        """
        Scales the data using MinMaxScaler.

        Returns:
        numpy.ndarray: The scaled data.
        """
        all_features = self.features
        self.scaled_data = self.scaler.fit_transform(self.data[all_features])
        self.scaled_data[np.isnan(self.scaled_data)] = 0
        return self.scaled_data

    def apply_pca(self):
        """
        Applies PCA to the scaled data.

        Returns:
        numpy.ndarray: The transformed data after applying PCA.
        """
        if self.n_components > len(self.features):
            raise ValueError(f"n_components cannot be greater than the number of features ({len(self.features)}).")

        self.scaled_data = self.pca.fit_transform(self.scaled_data)
        return self.scaled_data

    def prepare_datasets(self):
        """
        Prepares the training and testing datasets from the scaled data.

        Returns:
        tuple: A tuple containing the training and testing datasets (X_train, X_test, Y_train, Y_test).
        """
        processor = DatasetProcessor(time_step=self.time_step, train_ratio=self.train_ratio)
        self.X_train, self.X_test, self.Y_train, self.Y_test = processor.create_train_test_sets(self.scaled_data)
        return self.X_train, self.X_test, self.Y_train, self.Y_test

    def train_model(self):
        """
        Trains the model using the training datasets.

        Returns:
        None.
        """
        trainer = ModelTrainer(learning_rate=self.learning_rate, epochs=self.epochs,
                               layer_architecture=self.layer_architecture, batch_size=self.batch_size,
                               current_model=self.current_model, current_loss=self.current_loss,
                               current_optimizer=self.current_optimizer)
        self.model = trainer.fit(self.X_train, self.Y_train, self.X_test, self.Y_test)

    def save_model_and_scaler(self):
        """
        Saves the trained model, scaler, and PCA in-memory and uploads them to Google Drive.

        Parameters:
        folder_name (str): The Google Drive folder name to upload the files.
        model_name (str): The name of the model file to be saved.
        scaler_name (str): The name of the scaler file to be saved.
        pca_name (str): The name of the PCA file to be saved.

        Returns:
        None.
        """
        # Save the trained model, scaler, and PCA in-memory
        model_bytes = BytesIO()
        scaler_bytes = BytesIO()
        pca_bytes = BytesIO()

        joblib.dump(self.model, model_bytes)
        joblib.dump(self.scaler, scaler_bytes)
        joblib.dump(self.pca, pca_bytes)

        # Reset the buffer positions to the beginning
        model_bytes.seek(0)
        scaler_bytes.seek(0)
        pca_bytes.seek(0)

        if self.model_path is not None:
            logger.info(f"Uploading model, scaler, and PCA to Google Drive folder {self.model_path}...")
            self.google_drive_handler.upload_file_from_memory(model_bytes, self.model_path, self.model_name)
            self.google_drive_handler.upload_file_from_memory(scaler_bytes, self.model_path, self.scaler_name)
            self.google_drive_handler.upload_file_from_memory(pca_bytes, self.model_path, self.pca_name)
            logger.info("Model, scaler, and PCA successfully uploaded to Google Drive.")

    def run(self):
        """
        Runs the entire training pipeline.

        Returns:
        None.
        """
        self.load_data()
        self.create_features()
        self.scale_data()
        self.apply_pca()
        self.prepare_datasets()
        self.train_model()
        self.save_model_and_scaler()
