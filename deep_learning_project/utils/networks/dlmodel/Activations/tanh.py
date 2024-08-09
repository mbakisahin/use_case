import numpy as np

class Tanh:
    """
    Hyperbolic Tangent (Tanh) activation function.

    The Tanh activation function is defined as:
    f(x) = tanh(x)
    It maps any real-valued number to the range (-1, 1).
    """

    def forward(self, inputs):
        """
        Forward pass through the Tanh activation function.

        Parameters:
        inputs (numpy.ndarray): The input values to the activation function.

        Returns:
        None. The function stores the output in the instance variable `self.output`.
        """
        self.output = np.tanh(inputs)

    def backward(self, dvalues):
        """
        Backward pass through the Tanh activation function.

        Parameters:
        dvalues (numpy.ndarray): The gradient of the loss function with respect to the output of the Tanh function.

        Returns:
        None. The function stores the gradient with respect to the input in the instance variable `self.dinputs`.
        """
        self.dinputs = dvalues * (1 - self.output ** 2)
