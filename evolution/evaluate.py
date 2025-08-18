import numpy as np
from agents.neural_controller import NeuralController
from agents.drone import Drone
from environment.field import generate_field
from environment.pheromone_map import PheromoneMap
from simulation.metrics import compute_fitness
from simulation.loop import generate_start_positions

def evaluate_controller(weights, grid_size=(20, 20), num_drones=5, timesteps=200,
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
        # energy = np.mean([len(d.path) - 1 for d in drones if d.active])
        moves_per_drone = [max(0, len(d.path) - 1) for d in drones if d.active]
        energy = (np.mean(moves_per_drone) / timesteps) if moves_per_drone else 0.0

        explored_row_ratio = np.count_nonzero(np.sum(visit_map, axis=1)) / grid_size[0]
        explored_col_ratio = np.count_nonzero(np.sum(visit_map, axis=0)) / grid_size[1]

        # max_phero_ratio = np.max(pheromone.map) / (np.sum(pheromone.map) + 1e-6)
        visited = (visit_map > 0)
        mean_phero_visited = float(pheromone.map[visited].mean()) if visited.any() else 0.0

        late_exploration_bonus = coverage_progress[-1] - coverage_progress[int(0.5 * timesteps)]

        fmin, fmax = field.min(), field.max()
        health_norm = (field - fmin) / (fmax - fmin + 1e-9)
        stress = 1.0 - health_norm

        visited = (visit_map > 0)
        stress_covered = float((stress * visited).sum() / (stress.sum() + 1e-9))

        q = 0.90
        thresh = np.quantile(stress, q)
        hotspots = (stress >= thresh)
        hotspot_recall = float((visited & hotspots).sum() / (hotspots.sum() + 1e-9))

        coverage_auc = float(np.mean(coverage_progress))

        # Fitness
        fitness = (
            30.0 * coverage
            - 30.0 * overlap
            - 0.5 * energy
            + 1.5 * explored_row_ratio
            + 1.5 * explored_col_ratio
            + 1.5 * late_exploration_bonus
            - 20.0 * mean_phero_visited
            + 8.0 * np.sqrt(explored_row_ratio * explored_col_ratio)
            + 30.0 * stress_covered
            + 10.0 * hotspot_recall
            +10.0 * coverage_auc
        )
        fitnesses.append(fitness)

    return np.mean(fitnesses)
