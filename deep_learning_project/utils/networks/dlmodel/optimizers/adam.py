import numpy as np


class OptimizerAdam:
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        """
        Initialize the Adam optimizer with given hyperparameters.

        Parameters:
        learning_rate (float): The learning rate for the optimizer.
        beta_1 (float): The exponential decay rate for the first moment estimates.
        beta_2 (float): The exponential decay rate for the second moment estimates.
        epsilon (float): A small constant for numerical stability.
        """
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.iterations = 0

    def update_params(self, layer):
        """
        Update the parameters of the given layer using the Adam optimization algorithm.

        Parameters:
        layer (Layer): The layer whose parameters are to be updated.
        """
        if not hasattr(layer, 'weight_momentums'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update the iterations count
        self.iterations += 1

        # Update momentum with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        # Corrected momentum
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** self.iterations)
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** self.iterations)

        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases ** 2

        # Corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** self.iterations)
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** self.iterations)

        # Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights -= self.learning_rate * weight_momentums_corrected / (
                    np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases -= self.learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
