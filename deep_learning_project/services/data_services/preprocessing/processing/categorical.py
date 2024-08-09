from services.data_services.preprocessing.processing import DataPreprocessing

class CategoricalDataPreprocessing:
    """
    A class for processing categorical processing.

    This class initializes with a file path, uses a DataPreprocessing instance to preprocess the processing,
    and provides methods to preprocess specific columns and drop unnecessary columns.
    """

    def __init__(self, file_path, file_type):
        """
        Initializes the CategoricalDataPreprocessing with a file path.

        Parameters:
        file_path (str): The path to the processing file.

        Returns:
        None.
        """
        self.file_path = file_path
        self.data_preprocessor = DataPreprocessing(file_path, file_type)
        self.df = self.data_preprocessor.get_data()

    def preprocess_data(self):
        """
        Preprocesses the processing by performing necessary transformations on specified columns
        and then dropping unnecessary columns.

        Returns:
        pandas.DataFrame: The preprocessed dataframe.
        """
        self.df = self.data_preprocessor.preprocess_data(columns=['item_category_id'])
        self.df = self._drop_columns()
        return self.df

    def _drop_columns(self):
        """
        Drops unnecessary columns from the dataframe.

        Returns:
        pandas.DataFrame: The dataframe with specified columns dropped.
        """
        self.df = self.df.drop(columns=['item_category_name'])
        return self.df
