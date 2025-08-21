import numpy as np

class NeuralController:
    def __init__(self, input_size=29, hidden_size=32, output_size=5, weights=None,  x_mean=None, x_std=None):
        """
        A simple 1-hidden-layer neural network with ReLU activation.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # per-feature normalization
        self.x_mean = np.zeros(input_size) if x_mean is None else np.asarray(x_mean)
        self.x_std = np.ones(input_size) if x_std is None else np.maximum(np.asarray(x_std), 1e-6)

        # params
        self.n_params = (input_size * hidden_size) + hidden_size + \
                        (hidden_size * output_size) + output_size

        if weights is not None:
            self.set_weights(weights)
        else:
            self.init_he_weights()

    def init_he_weights(self):
        w1_scale = np.sqrt(2.0 / self.input_size)  # He init for ReLU
        w2_scale = np.sqrt(2.0 / self.hidden_size)
        self.W1 = np.random.randn(self.hidden_size, self.input_size) * w1_scale
        self.b1 = np.zeros((self.hidden_size,))
        self.W2 = np.random.randn(self.output_size, self.hidden_size) * w2_scale
        self.b2 = np.zeros((self.output_size,))

    def set_weights(self, flat_weights):
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
        return np.concatenate([
            self.W1.flatten(), self.b1,
            self.W2.flatten(), self.b2
        ])
    def _normalize(self, x):
        x = (x - self.x_mean) / self.x_std   # z-score
        return np.clip(x, -3.0, 3.0)

    def forward(self, x):
        x = self._normalize(x)
        z1 = self.W1 @ x + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        z2 = self.W2 @ a1 + self.b2
        return z2