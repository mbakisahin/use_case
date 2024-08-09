from services.data_services.preprocessing.processing import DataPreprocessing

class ShopDataPreprocessor:
    """
    A class for processing shop processing.

    This class initializes with a file path, uses a DataPreprocessing instance to preprocess the processing,
    and provides methods to preprocess specific columns and rename unnecessary columns.
    """

    def __init__(self, file_path, file_type):
        """
        Initializes the ShopDataPreprocessor with a file path.

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
        and then renaming and dropping unnecessary columns.

        Returns:
        pandas.DataFrame: The preprocessed dataframe.
        """
        # Perform DataPreprocessing steps
        self.df = self.data_preprocessor.preprocess_data(columns=['shop_id'])

        # Perform ShopDataPreprocessor steps
        self.df = self._rename_columns()
        return self.df

    def _rename_columns(self):
        """
        Renames and drops unnecessary columns from the dataframe.

        Returns:
        pandas.DataFrame: The dataframe with specified columns renamed and dropped.
        """
        self.df = self.df.rename(columns={'shop_id': 'shop'})
        self.df = self.df.drop(columns=['shop_name'])
        return self.df
