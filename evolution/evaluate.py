import numpy as np
from agents.neural_controller import NeuralController
from agents.drone import Drone
from environment.field import generate_field
from environment.pheromone_map import PheromoneMap
from simulation.metrics import compute_fitness
from simulation.loop import generate_start_positions

# def generate_start_positions(grid_size, num_drones):
#     """
#     Evenly spread initial positions in the field to promote exploration.
#     """
#     positions = []
#     step_x = grid_size[0] // (num_drones + 1)
#     step_y = grid_size[1] // (num_drones + 1)
#
#     for i in range(1, num_drones + 1):
#         x = step_x * i
#         y = step_y * i
#         positions.append((x, y))
#
#     return positions

def evaluate_controller(weights, grid_size=(20, 20), num_drones=5, timesteps=100,
                        seeds=[42, 17, 73, 101, 123, 256, 512, 12, 65, 15]):
    """
    Evaluate a neural controller by averaging fitness over multiple seeds.
    Strongly rewards coverage and penalizes clustering.
    """
    fitnesses = []

    for seed in seeds:
        np.random.seed(seed)

        field = generate_field(*grid_size, seed=seed)
        pheromone = PheromoneMap(grid_size, decay_rate=0.01)
        visit_map = np.zeros(grid_size, dtype=int)

        controller = NeuralController(weights=weights)

        # Init drones
        start_positions = generate_start_positions(grid_size, num_drones)
        drones = [Drone(x, y, grid_size, controller=controller, visit_map=visit_map)
                  for (x, y) in start_positions]

        # Track coverage over time
        coverage_progress = []

        for t in range(timesteps):
            active_drones = [d for d in drones if d.active]
            occupied_positions = {(d.x, d.y) for d in active_drones}
            all_positions = [(d.x, d.y) for d in active_drones]

            for drone in drones:
                drone.maybe_fail()
                drone.decide_and_move(field, pheromone.map, occupied_positions, all_positions)
                if drone.active:
                    visit_map[drone.x, drone.y] += 1
                drone.deposit_pheromone(pheromone)

            pheromone.update()

            # Save coverage progression
            coverage_progress.append(np.count_nonzero(visit_map) / visit_map.size)

        # Metrics
        coverage = np.count_nonzero(visit_map) / visit_map.size
        total_visits = np.sum(visit_map)
        unique_visits = np.count_nonzero(visit_map)
        overlap = (total_visits - unique_visits) / total_visits if total_visits > 0 else 0
        energy = np.mean([len(d.path) - 1 for d in drones if d.active])

        explored_row_ratio = np.count_nonzero(np.sum(visit_map, axis=1)) / grid_size[0]
        explored_col_ratio = np.count_nonzero(np.sum(visit_map, axis=0)) / grid_size[1]

        # New: Penalize pheromone concentration
        max_phero_ratio = np.max(pheromone.map) / (np.sum(pheromone.map) + 1e-6)

        # New: Reward for still finding new cells near the end
        late_exploration_bonus = coverage_progress[-1] - coverage_progress[int(0.5 * timesteps)]

        # Fitness: coverage dominates
        fitness = (
            5.0 * coverage
            - 5.0 * overlap
            - 0.5 * energy
            + 1.0 * explored_row_ratio
            + 1.0 * explored_col_ratio
            + 2.0 * late_exploration_bonus
            - 5.0 * max_phero_ratio
        )
        fitnesses.append(fitness)

    return np.mean(fitnesses)
