from services.data_services.preprocessing.merge.merge import DataPreparer
from services.data_services.preprocessing.feature_engineering.engineering import FeatureEngineer
from utils.log.logger import get_logger

logger = get_logger(__name__)


if __name__ == "__main__":
    """
    Main script for preparing, processing, selecting features, and saving the data.

    This script prepares and processes the data, selects high correlation features,
    filters the data based on these features, and saves the filtered data to a CSV file.
    """

    logger.info("Starting the main script...")

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

    logger.info("Main script completed.")