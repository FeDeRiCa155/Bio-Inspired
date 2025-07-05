import matplotlib.pyplot as plt
import numpy as np

def plot_field_with_paths(field, drones, show_start=True, show_end=True):
    """
    Plot the field and the paths of all drones.

    Args:
        field (np.ndarray): 2D crop health map.
        drones (list): list of Drone objects.
        show_start (bool): mark start of drones.
        show_end (bool): mark end of drones.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(field, cmap='Greens', interpolation='nearest')
    plt.colorbar(label='Crop Health')
    plt.title('Drone Paths over Crop Field')
    plt.xlabel('Column')
    plt.ylabel('Row')

    for drone in drones:
        if not drone.active:
            fail_pos = np.array(drone.path[-1])  # Last known position
            plt.scatter(fail_pos[1], fail_pos[0], c='black', marker='x', s=40,
                        label='Failed' if 'Failed' not in plt.gca().get_legend_handles_labels()[1] else "")

        path = np.array(drone.path)
        plt.plot(path[:, 1], path[:, 0], lw=1, alpha=0.8)

        if show_start:
            plt.scatter(path[0, 1], path[0, 0], c='blue', s=20, label='Start' if 'Start' not in plt.gca().get_legend_handles_labels()[1] else "")
        if show_end:
            plt.scatter(path[-1, 1], path[-1, 0], c='red', s=20, label='End' if 'End' not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.show()

def plot_pheromone_map(pheromone_map):
    plt.figure(figsize=(8, 8))
    plt.imshow(pheromone_map, cmap='plasma', interpolation='nearest')
    plt.colorbar(label='Pheromone Level')
    plt.title('Final Pheromone Map')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.show()

def plot_visit_heatmap(visit_map):
    plt.figure(figsize=(8, 8))
    plt.imshow(visit_map, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Visit Count')
    plt.title('Field Visit Heatmap')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.show()
