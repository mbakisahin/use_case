import numpy as np

class Sigmoid:
    """
    Sigmoid activation function.

    The Sigmoid activation function is defined as:
    f(x) = 1 / (1 + exp(-x))
    It maps any real-valued number to the range (0, 1).
    """

    def forward(self, inputs):
        """
        Forward pass through the Sigmoid activation function.

        Parameters:
        inputs (numpy.ndarray): The input values to the activation function.

        Returns:
        None. The function stores the output in the instance variable `self.output`.
        """
        # Clipping the inputs to prevent overflow
        self.output = 1 / (1 + np.exp(-np.clip(inputs, -709, 709)))

    def backward(self, dvalues):
        """
        Backward pass through the Sigmoid activation function.

        Parameters:
        dvalues (numpy.ndarray): The gradient of the loss function with respect to the output of the Sigmoid function.

        Returns:
        None. The function stores the gradient with respect to the input in the instance variable `self.dinputs`.
        """
        self.dinputs = dvalues * (1 - self.output) * self.output
