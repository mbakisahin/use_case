from services.train_service.training_pipeline.data import DatasetCreator

class DatasetProcessor:
    """
    A class for processing datasets for training and testing.

    This class provides methods to create training and testing datasets from scaled processing,
    using a specified time step and training ratio.
    """

    def __init__(self, time_step=10, train_ratio=0.8):
        """
        Initializes the DatasetProcessor with specified time step and training ratio.

        Parameters:
        time_step (int, optional): The number of time steps to use for creating the dataset. Default is 10.
        train_ratio (float, optional): The ratio of the dataset to use for training. Default is 0.8.

        Returns:
        None.
        """
        self.time_step = time_step
        self.train_ratio = train_ratio
        self.dataset_creator = DatasetCreator()

    def create_train_test_sets(self, scaled_data):
        """
        Creates training and testing datasets from scaled processing.

        Parameters:
        scaled_data (numpy.ndarray): The scaled processing to create the dataset from.

        Returns:
        tuple: A tuple containing four numpy arrays:
               - X_train (numpy.ndarray): The training input processing.
               - X_test (numpy.ndarray): The testing input processing.
               - Y_train (numpy.ndarray): The training output processing.
               - Y_test (numpy.ndarray): The testing output processing.
        """
        X, Y = self.dataset_creator.create_dataset(scaled_data)

        train_size = int(len(X) * self.train_ratio)
        X_train, X_test = X[:train_size], X[train_size:]
        Y_train, Y_test = Y[:train_size, 0], Y[train_size:, 0]
        return X_train, X_test, Y_train.reshape(-1, 1), Y_test.reshape(-1, 1)
