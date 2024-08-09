from services.data_services.preprocessing.merge.merge import DataPreparer
from services.data_services.preprocessing.feature_engineering.engineering import FeatureEngineer
from utils.log.logger import get_logger

logger = get_logger(__name__)

class DataProcessingService:
    """
    Service for processing data including preparation, feature engineering,
    and saving/uploading processed data.
    """

    def process_data(self):
        """
        Process data by preparing it, engineering features, and saving/uploading.

        Returns:
            str: Status message indicating completion.
        """
        data_handler = DataPreparer(service='SERVICE_ACCOUNT_FILE',
                                    transaction='TRANSACTION_FILE_ID',
                                    category='CATEGORY_FILE_ID',
                                    item='ITEM_FILE_ID',
                                    shop='SHOP_FILE_ID',
                                    load_data='UPLOAD_DRIVE_FOLDER',
                                    data_name='SAVED_MERGED')

        # Prepare and process the data
        logger.info("Preparing data...")
        data = data_handler.prepare_data()
        logger.info("Data prepared.")

        engineer = FeatureEngineer(data)

        logger.info("Engineering features...")
        df = engineer.engineer_features()
        logger.info("Features engineered.")

        logger.info(f"Saving and uploading data to Google Drive...")
        data_handler.save_and_upload_data(df)
        logger.info("Data saved and uploaded.")

        return "Data processing completed."
