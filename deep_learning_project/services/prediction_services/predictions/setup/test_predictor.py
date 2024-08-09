import os

import pandas as pd

from utils.db import GoogleDriveHandler
from services.prediction_services.predictions.setup.predict import AmountPredictor
from utils.log.logger import get_logger

logger = get_logger(__name__)


class TestPredictor:
    def __init__(self, data_path, test_path, service, model_file_name, scaler_file_name, pca_file_name,
                 output_path, output_id):

        self.google_drive_handler = GoogleDriveHandler(service)

        self.data_path = os.getenv(data_path)
        self.test_path = os.getenv(test_path)
        self.service = service
        self.model_file_name = model_file_name
        self.scaler_file_name = scaler_file_name
        self.pca_file_name = pca_file_name
        self.output_path = os.getenv(output_path)
        self.output_id = os.getenv(output_id)
        self.predictor = None
        self.df = None
        self.testdf = None

    def load_data(self):
        """
        Loads the data from the CSV file.

        Returns:
        pandas.DataFrame: The loaded data.
        """

        file_id = self.google_drive_handler.search_file_by_name(self.data_path)

        downloaded_file = self.google_drive_handler.download_file_from_drive(file_id)
        self.df = pd.read_csv(downloaded_file)

        downloaded_test_file = self.google_drive_handler.download_file_from_drive(self.test_path)
        self.testdf = pd.read_csv(downloaded_test_file)

    def initialize_predictor(self):
        self.predictor = AmountPredictor(
            service=self.service,
            file_path=self.df,
            model_file_name=self.model_file_name,
            scaler_file_name=self.scaler_file_name,
            pca_file_name=self.pca_file_name
        )
        self.predictor.run()

    def make_predictions(self):
        target = 'amount'
        features = [col for col in self.df.columns if col != target]
        unique_combinations = self.testdf[['shop', 'item']].drop_duplicates()

        predictions = []

        for index, row in unique_combinations.iterrows():
            shop_id = row['shop']
            item_id = row['item']
            try:
                prediction = self.predictor.predict(shop_id, item_id, features)
                logger.info(f'shop_id: {shop_id}, item_id: {item_id}, Predicted amount: {prediction}')
                predictions.append({
                    'shop_id': shop_id,
                    'item_id': item_id,
                    'predicted_amount': prediction
                })
            except (IndexError, ValueError) as e:
                logger.info(f'Not enough data for shop_id: {shop_id}, item_id: {item_id}: {e}')

        # Predictions list to DataFrame
        predictions_df = pd.DataFrame(predictions)

        return predictions_df


    def make_api_prediction(self, shop_id, item_id):
        target = 'amount'
        features = [col for col in self.df.columns if col != target]

        prediction = self.predictor.predict(shop_id, item_id, features)
        logger.info(f'shop_id: {shop_id}, item_id: {item_id}, Predicted amount: {prediction}')


        return prediction


    def save_predictions(self, dataframe):
        dataframe.to_csv(self.output_path, index=False)
        self.google_drive_handler.upload_file_to_drive(self.output_path, self.output_id)
        os.remove(self.output_path)


    def run(self):
        self.load_data()
        self.initialize_predictor()
        # predictions = self.make_predictions()
        # self.save_predictions(predictions)
