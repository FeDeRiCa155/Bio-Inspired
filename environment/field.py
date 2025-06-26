import numpy as np

def generate_field(rows, cols, seed=None):
    """
    Generate a 2D crop field. The crops are:
    - Healthy: 1.0
    - Moderately unhealthy: 0.6
    - Severely unhealthy: 0.2

    Args:
        rows (int): Number of rows in the field.
        cols (int): Number of columns in the field.
        seed (int, optional): Random seed.

    Returns:
        np.ndarray: 2D array of shape (rows, cols) with health values.
    """
    if seed is not None:
        np.random.seed(seed)

    field = np.ones((rows, cols))

    # Health percentages
    total_cells = rows * cols
    severe_ratio = 0.1
    moderate_ratio = 0.15

    num_severe = int(severe_ratio * total_cells)
    num_moderate = int(moderate_ratio * total_cells)

    severe_indices = np.random.choice(total_cells, num_severe, replace=False)
    remaining_indices = list(set(range(total_cells)) - set(severe_indices))
    moderate_indices = np.random.choice(remaining_indices, num_moderate, replace=False)

    # Apply health levels
    for idx in severe_indices:
        i, j = divmod(idx, cols)
        field[i, j] = 0.2

    for idx in moderate_indices:
        i, j = divmod(idx, cols)
        field[i, j] = 0.6

    return field


def display_field(field):
    """
    Display the field using matplotlib.

    Args:
        field (np.ndarray): 2D crop health map
    """
    import matplotlib.pyplot as plt

    plt.imshow(field, cmap='Greens', interpolation='nearest')
    plt.colorbar(label='Crop Health')
    plt.title('Crop Field')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.show()


# Example usage
if __name__ == "__main__":
    test_field = generate_field(rows=30, cols=30, seed=42)
    display_field(test_field)
