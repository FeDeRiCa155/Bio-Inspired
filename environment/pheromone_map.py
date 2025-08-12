import numpy as np

class PheromoneMap:
    """
    Generate a pheromone map.

    Args:
        shape (tuple): (rows, cols) size of the field.
        decay_rate (float): percentage of pheromone to evaporate per time step.
    """

    def __init__(self, shape, decay_rate=0.08, max_pheromone=5.0, diffusion_alpha=0.10):
        self.map = np.zeros(shape, dtype=np.float32)
        self.decay_rate = float(decay_rate)
        self.max_pheromone = float(max_pheromone)
        self.diffusion_alpha = float(diffusion_alpha)
        self.map = np.zeros(shape, dtype=np.float32)

    def deposit(self, x, y, amount=0.5):
        """
        Deposit pheromone at a specific cell location.

        Args:
            x (int): row index.
            y (int): column index.
            amount (float): how much pheromone to add.
        """
        self.map[x, y] = min(self.map[x, y] + amount, self.max_pheromone)

    def update(self):
        """
        Apply decay to the pheromone map.
        """
        self.map *= (1.0 - self.decay_rate)
        if self.diffusion_alpha > 0.0:
            blurred = self._diffuse(self.map)
            self.map = (1.0 - self.diffusion_alpha) * self.map + self.diffusion_alpha * blurred

    def get_value(self, x, y):
        """
        Get pheromone value at a cell.

        Args:
            x (int): row index.
            y (int): column index.

        Returns:
            float: pheromone level.
        """
        return self.map[x, y]

    def get_local_view(self, x, y, radius=1):
        """
        Get a local patch of the pheromone map around the agent.

        Args:
            x (int): row index.
            y (int): column index.
            radius (int): number of cells in each direction.

        Returns:
            np.ndarray: (2*radius+1, 2*radius+1) grid centered on (x, y).
        """
        x_min = max(x - radius, 0)
        x_max = min(x + radius + 1, self.map.shape[0])
        y_min = max(y - radius, 0)
        y_max = min(y + radius + 1, self.map.shape[1])

        patch = self.map[x_min:x_max, y_min:y_max]
        patch += np.random.normal(0, 0.01, patch.shape)
        pad_x = (max(0, radius - x), max(0, x + radius + 1 - self.map.shape[0]))
        pad_y = (max(0, radius - y), max(0, y + radius + 1 - self.map.shape[1]))

        local = np.pad(patch, (pad_x, pad_y), mode='constant', constant_values=0.0)

        return local


    @staticmethod
    def _diffuse(arr):
        pad = np.pad(arr, 1, mode='edge')
        acc = (
            pad[0:-2,0:-2] + pad[0:-2,1:-1] + pad[0:-2,2:] +
            pad[1:-1,0:-2] + pad[1:-1,1:-1] + pad[1:-1,2:] +
            pad[2:  ,0:-2] + pad[2:  ,1:-1] + pad[2:  ,2:]
        ) / 9.0
        return acc.astype(arr.dtype, copy=False)

    def reset(self):
        """
        Reset the pheromone map to all zeros.
        """
        self.map.fill(0.0)
