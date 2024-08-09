import numpy as np
from utils.log.logger import get_logger

logger = get_logger(__name__)



class ModelTrainer():
    """
    A class for building, training, and evaluating a deep learning model.

    This class provides methods to build a model with specified architecture,
    train it on provided processing, and evaluate its performance on training and testing datasets.
    """

    def __init__(self, learning_rate, epochs, layer_architecture, batch_size,
                 current_model, current_loss, current_optimizer):
        """
        Initializes the ModelTrainer with specified learning rate, number of epochs, and layer architecture.

        Parameters:
        learning_rate (float): The learning rate for the optimizer.
        epochs (int): The number of epochs to train the model.
        layer_architecture (list): The architecture of the layers.

        Returns:
        None.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.layer_architecture = layer_architecture
        self.batch_size = batch_size
        self.current_model = current_model
        self.current_loss = current_loss
        self.current_optimizer = current_optimizer


    def build_model(self, input_size):
        """
        Builds a deep learning model with a specified architecture.

        Parameters:
        input_size (int): The number of input features.

        Returns:
        DPModel: The constructed deep learning model.
        """
        # model = self.current_model.copy()
        self.add_layers(self.current_model, input_size)
        self.current_model.set_loss(self.current_loss)
        self.current_model.set_optimizer(self.current_optimizer(learning_rate=self.learning_rate))
        return self.current_model

    def add_layers(self, model, input_size):
        """
        Adds layers to the model.

        Parameters:
        model (DPModel): The model to which layers will be added.
        input_size (int): The number of input features.

        Returns:
        None.
        """
        current_input_size = input_size
        for layer in self.layer_architecture:
            self.current_model.add_layer(input_size=current_input_size,
                            output_size=layer['output_size'],
                            activation=layer.get('activation', None))
            current_input_size = layer['output_size']



    def batch_predict(self, model, X, batch_size=32):
        """
        Predicts the output for the given input processing in batches.

        Parameters:
        model (DPModel): The trained deep learning model.
        X (numpy.ndarray): The input processing to predict.
        batch_size (int): The size of the batches to use for predictions.

        Returns:
        numpy.ndarray: The predicted output processing.
        """
        n_samples = X.shape[0]
        predictions = []

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_predictions = model.predict(X[start:end])
            predictions.extend(batch_predictions)

        return np.array(predictions)

    def fit(self, X_train, Y_train, X_test, Y_test):
        """
        Trains the model on the training processing and evaluates it on the testing processing.

        Parameters:
        X_train (numpy.ndarray): The training input processing.
        Y_train (numpy.ndarray): The training output processing.
        X_test (numpy.ndarray): The testing input processing.
        Y_test (numpy.ndarray): The testing output processing.

        Returns:
        DPModel: The trained deep learning model.
        """
        model = self.build_model(X_train.shape[1])
        model.train(X_train, Y_train, epochs=self.epochs, batch_size=self.batch_size)

        # Performance on training processing
        train_predictions = self.batch_predict(model, X_train)
        train_loss = model.loss.forward(y_pred=train_predictions, y_true=Y_train)
        train_rmse = np.sqrt(train_loss)
        logger.info(f'Train RMSE: {train_rmse}')

        # Performance on testing processing
        test_predictions = self.batch_predict(model, X_test)
        test_loss = model.loss.forward(y_pred=test_predictions, y_true=Y_test)
        test_rmse = np.sqrt(test_loss)
        logger.info(f'Test RMSE: {test_rmse}')

        return model

