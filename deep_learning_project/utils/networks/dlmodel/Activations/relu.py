import numpy as np

class ReLU:
    """
    Rectified Linear Unit (ReLU) activation function.

    The ReLU activation function is defined as:
    f(x) = max(0, x)
    It sets all negative input values to zero while keeping positive values unchanged.
    """

    def forward(self, inputs):
        """
        Forward pass through the ReLU activation function.

        Parameters:
        inputs (numpy.ndarray): The input values to the activation function.

        Returns:
        None. The function stores the output in the instance variable `self.output`.
        """
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        """
        Backward pass through the ReLU activation function.

        Parameters:
        dvalues (numpy.ndarray): The gradient of the loss function with respect to the output of the ReLU function.

        Returns:
        None. The function stores the gradient with respect to the input in the instance variable `self.dinputs`.
        """
        self.dinputs = dvalues.copy()
        self.dinputs[self.output <= 0] = 0
