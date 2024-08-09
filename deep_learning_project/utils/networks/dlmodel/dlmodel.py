import numpy as np

from utils.networks.dlmodel import Layer

from utils.log.logger import get_logger

logger = get_logger(__name__)



class DPModel:
    def __init__(self):
        """
        Initialize the DPModel with empty lists for layers and activations,
        and set loss and optimizer to None.
        """
        self.layers = []
        self.activations = []
        self.loss = None
        self.optimizer = None

    def add_layer(self, input_size, output_size, activation=None):
        """
        Add a layer to the model.

        Parameters:
        input_size (int): Number of input features to the layer.
        output_size (int): Number of neurons in the layer.
        activation (callable, optional): Activation function to be applied to the layer's output.
        """
        layer = Layer(input_size, output_size)
        self.layers.append(layer)
        if activation:
            self.activations.append(activation())

    def set_loss(self, loss):
        """
        Set the loss function for the model.

        Parameters:
        loss (object): Loss function object with forward and backward methods.
        """
        self.loss = loss

    def set_optimizer(self, optimizer):
        """
        Set the optimizer for the model.

        Parameters:
        optimizer (object): Optimizer object with an update_params method.
        """
        self.optimizer = optimizer

    def forward(self, inputs):
        """
        Perform a forward pass through the model.

        Parameters:
        inputs (ndarray): Input data.

        Returns:
        ndarray: Output of the model.
        """
        output = inputs
        for i in range(len(self.layers)):
            self.layers[i].forward(output)
            output = self.layers[i].output
            if i < len(self.activations):
                self.activations[i].forward(output)
                output = self.activations[i].output
        return output

    def backward(self, output, y_true):
        """
        Perform a backward pass through the model.

        Parameters:
        output (ndarray): Output of the model.
        y_true (ndarray): True target values.
        """
        self.loss.backward(output, y_true)
        dvalues = self.loss.dinputs
        for i in reversed(range(len(self.layers))):
            if i < len(self.activations):
                self.activations[i].backward(dvalues)
                dvalues = self.activations[i].dinputs
            self.layers[i].backward(dvalues)
            dvalues = self.layers[i].dinputs

    def create_batches(self, X, y, batch_size):
        """
        Create batches of data from the given inputs and targets.

        Parameters:
        X (ndarray): Input data.
        y (ndarray): Target values.
        batch_size (int): Number of samples per batch.

        Returns:
        list: List of tuples, each containing a batch of input data and target values.
        """
        num_samples = X.shape[0]
        indices = np.random.permutation(num_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        num_batches = int(np.ceil(num_samples / batch_size))
        batches = [
            (X_shuffled[i * batch_size:(i + 1) * batch_size], y_shuffled[i * batch_size:(i + 1) * batch_size])
            for i in range(num_batches)
        ]
        return batches

    def train(self, X, y, epochs=1, batch_size=1):
        """
        Train the model using the given data.

        Parameters:
        X (ndarray): Input data.
        y (ndarray): Target values.
        epochs (int, optional): Number of epochs to train for. Defaults to 1.
        batch_size (int, optional): Number of samples per batch. Defaults to 1.
        """
        for epoch in range(epochs):
            batches = self.create_batches(X, y, batch_size)
            epoch_loss = 0
            for batch_X, batch_y in batches:
                output = self.forward(batch_X)
                loss = self.loss.forward(y_pred=output, y_true=batch_y)
                self.backward(output, batch_y)
                for layer in self.layers:
                    self.optimizer.update_params(layer)
                epoch_loss += loss
            epoch_loss /= len(batches)
            rmse = np.sqrt(epoch_loss)
            logger.info(f'Epoch {epoch + 1}, Loss: {epoch_loss}, RMSE: {rmse}')

    def predict(self, inputs):
        """
        Predict the outputs for the given inputs.

        Parameters:
        inputs (ndarray): Input data.

        Returns:
        ndarray: Predicted values.
        """
        return self.forward(inputs)
