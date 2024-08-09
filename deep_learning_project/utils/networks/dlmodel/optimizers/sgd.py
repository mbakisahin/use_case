import numpy as np


class OptimizerSGD:
    def __init__(self, learning_rate=1.0):
        """
        Initialize the SGD optimizer with a given learning rate.

        Parameters:
        learning_rate (float): The learning rate for the optimizer.
        """
        self.learning_rate = learning_rate

    def update_params(self, layer):
        """
        Update the parameters of the given layer using the Stochastic Gradient Descent (SGD) optimization algorithm.

        Parameters:
        layer (Layer): The layer whose parameters are to be updated.
        """
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases
