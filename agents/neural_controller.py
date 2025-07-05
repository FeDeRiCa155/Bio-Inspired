import numpy as np

class NeuralController:
    def __init__(self, input_size=27, hidden_size=16, output_size=5, weights=None):
        """
        A simple 1-hidden-layer neural network with ReLU activation.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.n_params = (input_size * hidden_size) + hidden_size + (hidden_size * output_size) + output_size

        if weights is not None:
            self.set_weights(weights)
        else:
            self.init_random_weights()

    def init_random_weights(self):
        self.W1 = np.random.uniform(-1, 1, (self.hidden_size, self.input_size))
        self.b1 = np.random.uniform(-1, 1, (self.hidden_size,))
        self.W2 = np.random.uniform(-1, 1, (self.output_size, self.hidden_size))
        self.b2 = np.random.uniform(-1, 1, (self.output_size,))

    def set_weights(self, flat_weights):
        """
        Load weights from a flat vector.
        """
        assert len(flat_weights) == self.n_params
        idx = 0
        self.W1 = flat_weights[idx:idx + self.hidden_size * self.input_size].reshape(self.hidden_size, self.input_size)
        idx += self.hidden_size * self.input_size
        self.b1 = flat_weights[idx:idx + self.hidden_size]
        idx += self.hidden_size
        self.W2 = flat_weights[idx:idx + self.output_size * self.hidden_size].reshape(self.output_size, self.hidden_size)
        idx += self.output_size * self.hidden_size
        self.b2 = flat_weights[idx:idx + self.output_size]

    def get_weights(self):
        """
        Return flat weight vector for evolution.
        """
        return np.concatenate([
            self.W1.flatten(), self.b1,
            self.W2.flatten(), self.b2
        ])

    def forward(self, x):
        """
        Compute output vector from input vector x.
        Args:
            x (np.ndarray): shape (27,)
        Returns:
            np.ndarray: shape (5,), action scores
        """
        z1 = self.W1 @ x + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        z2 = self.W2 @ a1 + self.b2
        return z2