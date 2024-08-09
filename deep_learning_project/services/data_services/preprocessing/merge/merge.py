import os

from utils.db.uploader import GoogleDriveHandler

from services.data_services.preprocessing.processing import ShopDataPreprocessor, ItemDataPreprocessing, CategoricalDataPreprocessing, TransactionDataPreprocessor

class DataPreparer:
    """
    A class for preparing and merging data from multiple sources.

    This class initializes with file paths to transaction, category, item, and shop data files,
    and provides a method to preprocess and merge these datasets.
    """

    def __init__(self, service, transaction, category, item, shop, load_data, data_name):
        """
        Initializes the DataPreparer with predefined file paths.

        Returns:
        None.
        """

        self.google_drive_handler = GoogleDriveHandler(service)

        self.transaction_file_id = os.getenv(transaction)
        self.category_file_id = os.getenv(category)
        self.item_file_id = os.getenv(item)
        self.shop_file_id = os.getenv(shop)
        self.load_file = os.getenv(load_data)
        self.data_name = os.getenv(data_name)



    def prepare_data(self):
        """
        Prepares and merges data from transaction, category, item, and shop files.

        Returns:
        pandas.DataFrame: The merged and processed dataframe.
        """

        transaction_file =  self.google_drive_handler.download_file_from_drive(self.transaction_file_id)
        category_file =  self.google_drive_handler.download_file_from_drive(self.category_file_id)
        shop_file =  self.google_drive_handler.download_file_from_drive(self.shop_file_id)
        item_file =  self.google_drive_handler.download_file_from_drive(self.item_file_id)

        transaction_list_processed_data = TransactionDataPreprocessor(transaction_file, 'transaction').preprocess_data()
        category_list_processed_data = CategoricalDataPreprocessing(category_file, 'category').preprocess_data()
        shop_list_processed_data = ShopDataPreprocessor(shop_file, 'shop').preprocess_data()
        item_list_processed_data = ItemDataPreprocessing(item_file, 'item').preprocess_data()

        merged_data = transaction_list_processed_data.merge(item_list_processed_data, on='item', how='left') \
                                  .merge(category_list_processed_data, on='item_category_id', how='left') \
                                  .merge(shop_list_processed_data, on='shop', how='left')

        return merged_data

    def save_and_upload_data(self, dataframe):
        """
        Saves the dataframe to a CSV file and uploads it to Google Drive.

        Args:
        dataframe (pandas.DataFrame): The dataframe to save and upload.
        file_path (str): The path where the CSV file will be saved.
        folder_id (str): The ID of the folder to upload the file to.

        Returns:
        None.
        """
        dataframe.to_csv(self.data_name, index=False)
        self.google_drive_handler.upload_file_to_drive(self.data_name, self.load_file)
        os.remove(self.data_name)
