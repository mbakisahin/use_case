import numpy as np

class DatasetCreator:
    """
    A class for creating datasets from sequential processing.

    This class provides a method to create input-output pairs from sequential processing.
    """

    def __init__(self):
        """
        Initializes the DatasetCreator.

        Returns:
        None.
        """
        pass

    def create_dataset(self, data):
        """
        Creates input-output pairs from sequential processing.

        Parameters:
        processing (numpy.ndarray or list): The sequential processing to create the dataset from.

        Returns:
        tuple: A tuple containing two numpy arrays:
               - X (numpy.ndarray): The input processing.
               - Y (numpy.ndarray): The output processing, which is the input processing shifted by one time step.
        """
        X, Y = [], []
        for i in range(len(data) - 1):
            X.append(data[i])
            Y.append(data[i + 1])
        return np.array(X), np.array(Y)
