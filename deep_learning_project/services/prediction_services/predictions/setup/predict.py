import os
import joblib
import pandas as pd
from utils.db.uploader import GoogleDriveHandler
from utils.log.logger import get_logger

logger = get_logger(__name__)

class AmountPredictor:
    """
    A class for predicting the amount using a trained model, scaler, and PCA.

    This class loads the data, model, scaler, and PCA from the provided file paths,
    and provides methods to calculate monthly averages and make predictions.
    """

    def __init__(self, service, file_path, model_file_name, scaler_file_name, pca_file_name):
        """
        Initializes the AmountPredictor with file paths for data, model, scaler, and PCA.

        Parameters:
        file_path (str): The path to the CSV data file.
        model_file_id (str): The ID of the trained model file on Google Drive.
        scaler_file_id (str): The ID of the scaler file on Google Drive.
        pca_file_id (str): The ID of the PCA file on Google Drive.
        gdrive_service (GoogleDriveHandler): Instance of GoogleDriveHandler for Google Drive operations.

        Returns:
        None.
        """
        self.google_drive_handler = GoogleDriveHandler(service)

        self.file_path = file_path
        self.model_file_name = os.getenv(model_file_name)
        self.scaler_file_name = os.getenv(scaler_file_name)
        self.pca_file_name = os.getenv(pca_file_name)
        self.data = None
        self.model = None
        self.scaler = None
        self.pca = None

    def load_data(self):
        """
        Loads the data from the CSV file.

        Returns:
        pandas.DataFrame: The loaded data.
        """

        self.data = self.file_path

    def load_model_and_scaler(self):
        """
        Loads the trained model, scaler, and PCA from Google Drive.

        Returns:
        None.
        """

        model_file_file_id = self.google_drive_handler.search_file_by_name(self.model_file_name)
        model_file = self.google_drive_handler.download_file_from_drive(model_file_file_id)

        scaler_file_file_id = self.google_drive_handler.search_file_by_name(self.scaler_file_name)
        scaler_file = self.google_drive_handler.download_file_from_drive(scaler_file_file_id)

        pca_file_file_id = self.google_drive_handler.search_file_by_name(self.pca_file_name)
        pca_file = self.google_drive_handler.download_file_from_drive(pca_file_file_id)

        # Load the model, scaler, and PCA from the downloaded files
        self.model = joblib.load(model_file)
        self.scaler = joblib.load(scaler_file)
        self.pca = joblib.load(pca_file)
        logger.info("Model, scaler, and PCA successfully loaded.")

    def get_monthly_avg(self, shop_id, item_id):
        """
        Calculates the monthly average values for the specified shop_id and item_id.

        Parameters:
        shop_id (int): The ID of the shop.
        item_id (int): The ID of the item.

        Returns:
        pandas.Series: The monthly average values.

        Raises:
        ValueError: If no data is found for the specified shop_id and item_id.
        """
        # Filter data by shop_id and item_id
        filtered_data = self.data[(self.data['shop'] == shop_id) & (self.data['item'] == item_id)]

        if filtered_data.empty:
            # If no data for shop_id and item_id combination, filter by item_id only
            filtered_data = self.data[self.data['item'] == item_id]

            if filtered_data.empty:
                # If no data for item_id only, filter by shop_id only
                filtered_data = self.data[self.data['shop'] == shop_id]

                if filtered_data.empty:
                    # If no data for shop_id only, raise an error
                    raise ValueError(f"No data found for: shop_id={shop_id}, item_id={item_id}")

        # Check for NaN values and fill with mean if any
        if filtered_data.isnull().values.any():
            filtered_data = filtered_data.fillna(filtered_data.mean())

        # Select only numeric columns for calculating mean
        numeric_cols = filtered_data.select_dtypes(include='number').columns
        mean_values = filtered_data[numeric_cols].mean()

        return mean_values

    def predict(self, shop_id, item_id, features):
        """
        Makes a predictions for the specified shop_id and item_id using the trained model.

        Parameters:
        shop_id (int): The ID of the shop.
        item_id (int): The ID of the item.
        features (list): The list of feature names to use for predictions.

        Returns:
        float: The predicted amount.
        """
        monthly_avg = self.get_monthly_avg(shop_id, item_id)

        # Ensure features are numeric
        numeric_features = [feature for feature in features if feature in monthly_avg.index]

        target_features = monthly_avg[numeric_features].values.reshape(1, -1)

        # Convert the features to a dataframe compatible with the scaler
        target_features_df = pd.DataFrame(target_features, columns=numeric_features)

        # Scale and transform the features using PCA
        scaled_features = self.scaler.transform(target_features_df)
        scaled_features_pca = self.pca.transform(scaled_features)

        # Make a predictions using the model
        prediction = self.model.predict(scaled_features_pca)

        return prediction[0]

    def run(self):
        """
        Runs the data loading and model, scaler, and PCA loading processes.

        Returns:
        None.
        """
        self.load_data()
        self.load_model_and_scaler()
