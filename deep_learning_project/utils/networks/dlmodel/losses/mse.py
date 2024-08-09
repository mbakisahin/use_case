import numpy as np


class LossMSE:
    def forward(self, y_pred, y_true):
        """
        Calculate the Mean Squared Error (MSE) loss.

        Parameters:
        y_pred (ndarray): Predicted values.
        y_true (ndarray): True values.

        Returns:
        float: Computed MSE loss.
        """
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, y_pred, y_true):
        """
        Calculate the gradient of the MSE loss with respect to the predictions.

        Parameters:
        y_pred (ndarray): Predicted values.
        y_true (ndarray): True values.

        Sets:
        self.dinputs (ndarray): Gradient of the loss with respect to the inputs.
        """
        samples = len(y_pred)
        outputs = y_pred.shape[1]
        self.dinputs = 2 * (y_pred - y_true) / outputs
        self.dinputs = self.dinputs / samples
