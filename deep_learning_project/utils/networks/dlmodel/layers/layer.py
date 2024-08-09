import numpy as np


class Layer:
    def __init__(self, input_size, output_size):
        """
        Initialize the layer with random weights and zero biases.

        Parameters:
        input_size (int): Number of input features.
        output_size (int): Number of neurons in the layer.
        """
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        self.output = None
        self.dinputs = None

    def forward(self, inputs):
        """
        Perform the forward pass.

        Parameters:
        inputs (ndarray): Input data.
        """
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        """
        Perform the backward pass.

        Parameters:
        dvalues (ndarray): Gradient of the loss with respect to the layer's outputs.
        """
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
