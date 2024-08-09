# services/predictor_service.py
from utils.log.logger import get_logger
from services.prediction_services.predictions.predict import TestPredictor

logger = get_logger(__name__)

class PredictorService:
    """
    Service for handling predictions.
    """

    def __init__(self):
        """
        Initialize the PredictorService with the necessary configuration.
        """
        self.test_predictor = TestPredictor(data_path='LAST_FILE_NAME',
                                            test_path='TEST_FILE_ID',
                                            service='SERVICE_ACCOUNT_FILE',
                                            model_file_name='MODEL_NAME',
                                            scaler_file_name='SCALER_NAME',
                                            pca_file_name='PCA_NAME',
                                            output_id='UPLOAD_DRIVE_FOLDER',
                                            output_path='PREDICTED_AMOUNT')
        logger.info("Initialized PredictorService")

    def run(self):
        """
        Run the prediction process.
        """
        self.test_predictor.run()
        logger.info("PredictorService run completed")

    def make_prediction(self, shop_id: int, item_id: int):
        """
        Make a prediction for the given shop and item.

        Args:
            shop_id (int): The ID of the shop.
            item_id (int): The ID of the item.

        Returns:
            float: The predicted value.
        """
        prediction = self.test_predictor.make_api_prediction(shop_id=shop_id, item_id=item_id)
        prediction_value = float(prediction[0])
        logger.info(f"Prediction made: {prediction_value}")
        return prediction_value
