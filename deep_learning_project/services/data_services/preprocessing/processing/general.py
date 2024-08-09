import re
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder
import io

class DataPreprocessing:
    """
    A class for performing data processing tasks on a dataset.

    This class provides methods for handling missing values, removing unnamed columns,
    removing outliers, encoding categorical data, and processing specific columns for item category and shop name.
    """

    def __init__(self, file, file_type=None):
        """
        Initializes the DataPreprocessing with a file.

        Parameters:
        file (str or io.BytesIO): The path to the data file or a BytesIO object.
        file_type (str): The type of the file, used to apply specific processing steps.

        Returns:
        None.
        """
        if isinstance(file, str):
            self.df = pd.read_csv(file)
            self.file_path = file
        else:
            self.df = pd.read_csv(file)
            self.file_path = None  # file_path, BytesIO olduğunda geçerli değil
        self.file_type = file_type
        self.original_shape = self.df.shape

    def handle_missing_values(self):
        """
        Handles missing values by dropping rows with any missing values.

        Returns:
        None.
        """
        self.df.dropna(inplace=True)

    def remove_unnamed_columns(self):
        """
        Removes columns with 'Unnamed' in their names.

        Returns:
        None.
        """
        unnamed_columns = [col for col in self.df.columns if 'Unnamed' in col]
        self.df.drop(columns=unnamed_columns, inplace=True)

    def remove_outliers(self, columns):
        """
        Removes outliers from the specified columns using z-score.

        Parameters:
        columns (list): The list of columns to check for outliers.

        Returns:
        None.
        """
        selected_columns = self.df[columns]
        z_scores = np.abs(zscore(selected_columns))
        self.df = self.df[(z_scores < 3).all(axis=1)]

    def encode_categorical_data(self):
        """
        Encodes categorical data using label encoding.

        Returns:
        None.
        """
        date_column = 'date'
        categorical_columns = self.df.select_dtypes(include=['object']).columns
        categorical_columns = [col for col in categorical_columns if col != date_column]
        for col in categorical_columns:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])

    def process_item_category(self):
        """
        Processes the 'item_category_name' column to extract the main category.

        Returns:
        None.
        """
        self.df['item_category_name'] = self.df['item_category_name'].astype(str)
        self.df['main_category'] = self.df['item_category_name'].str.split(' - ').str[0]

    def process_shop_name(self):
        """
        Processes the 'shop_name' column to extract the country ID.

        Returns:
        None.
        """
        self.df['shop_name'] = self.df['shop_name'].astype(str)
        self.df['country_id'] = self.df['shop_name'].apply(lambda x: re.split(r'[ ,]', x)[0])

    def preprocess_data(self, columns):
        """
        Preprocesses the data by performing various processing tasks.

        Parameters:
        columns (list): The list of columns to check for outliers.

        Returns:
        pandas.DataFrame: The preprocessed dataframe.
        """
        self.remove_unnamed_columns()

        # Special processing for specific files
        if self.file_type == 'category':
            self.process_item_category()

        if self.file_type == 'shop':
            self.process_shop_name()

        self.encode_categorical_data()
        self.handle_missing_values()
        # self.remove_outliers(columns)
        return self.df

    def get_data(self):
        """
        Retrieves the processed dataframe.

        Returns:
        pandas.DataFrame: The processed dataframe.
        """
        return self.df
