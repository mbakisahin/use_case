from services.data_services.preprocessing.processing import DataPreprocessing

class TransactionDataPreprocessor:
    """
    A class for processing transaction data.

    This class initializes with a file, uses a DataPreprocessing instance to preprocess the data,
    and provides methods to calculate total sales and group the data by specific columns.
    """

    def __init__(self, file, file_type=None):
        """
        Initializes the TransactionDataPreprocessor with a file.

        Parameters:
        file (str or io.BytesIO): The path to the data file or a BytesIO object.
        file_type (str): The type of the file, used to apply specific processing steps.

        Returns:
        None.
        """
        self.file = file
        self.data_preprocessor = DataPreprocessing(file, file_type)
        self.df = self.data_preprocessor.get_data()

    def preprocess_data(self):
        """
        Preprocesses the data by performing necessary transformations on specified columns
        and then calculating total sales and grouping the data by year, month, shop, and item.

        Returns:
        pandas.DataFrame: The preprocessed dataframe.
        """
        # Perform DataPreprocessing steps
        self.df = self.data_preprocessor.preprocess_data(columns=['price'])

        # Perform TransactionDataPreprocessor steps
        self.df = self._calculate_total_sales()
        return self.df

    def _calculate_total_sales(self):
        """
        Calculates the total sales by multiplying price and amount.

        Returns:
        pandas.DataFrame: The dataframe with the total sales calculated.
        """
        self.df['total_price'] = self.df['price'] * self.df['amount']
        return self.df
