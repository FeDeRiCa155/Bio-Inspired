import numpy as np

class PheromoneMap:
    def __init__(self, shape, decay_rate=0.05):
        """
        Generate a pheromone map.

        Args:
            shape (tuple): (rows, cols) size of the field.
            decay_rate (float): percentage of pheromone to evaporate per time step.
        """
        self.map = np.zeros(shape, dtype=np.float32)
        self.decay_rate = decay_rate

    def deposit(self, x, y, amount=1.0):
        """
        Deposit pheromone at a specific cell location.

        Args:
            x (int): row index.
            y (int): column index.
            amount (float): how much pheromone to add.
        """
        self.map[x, y] += amount

    def update(self):
        """
        Apply decay to the pheromone map.
        """
        self.map *= (1.0 - self.decay_rate)

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

        local = self.map[x_min:x_max, y_min:y_max]

        return local

    def reset(self):
        """
        Reset the pheromone map to all zeros.
        """
        self.map.fill(0.0)
