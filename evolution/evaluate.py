import numpy as np
from agents.neural_controller import NeuralController
from agents.drone import Drone
from environment.field import generate_field
from environment.pheromone_map import PheromoneMap
from simulation.metrics import compute_fitness
from simulation.loop import generate_start_positions

def evaluate_controller(weights, grid_size=(20, 20), num_drones=5, timesteps=100,
                        seeds=[42, 17, 73, 101, 123, 256, 512, 12, 65, 15],
                        pheromone_decay=0.05, failure_prob=0.0):
    fitnesses = []

    for seed in seeds:
        np.random.seed(seed)

        field = generate_field(*grid_size, seed=seed)
        pheromone = PheromoneMap(grid_size, decay_rate=pheromone_decay)
        visit_map = np.zeros(grid_size, dtype=int)

        controller = NeuralController(weights=weights)

        # Init drones
        start_positions = generate_start_positions(grid_size, num_drones)
        drones = [Drone(x, y, grid_size, controller=controller,
                        visit_map=visit_map, failure_prob=failure_prob)
                  for (x, y) in start_positions]

        coverage_progress = []

        for t in range(timesteps):
            active = [d for d in drones if d.active]
            order = np.random.permutation(len(active))
            occupied = {(d.x, d.y) for d in active}
            all_pos  = [(d.x, d.y) for d in active]

            for k in order:
                d = active[k]
                d.maybe_fail()
                if not d.active:
                    continue

                old_xy = (d.x, d.y)
                occupied.discard(old_xy)

                d.decide_and_move(field, pheromone.map, occupied, all_pos)

                moved = (d.x, d.y) != old_xy
                if d.active and moved:
                    visit_map[d.x, d.y] += 1
                    d.deposit_pheromone(pheromone)

                occupied.add((d.x, d.y))
                all_pos.append((d.x, d.y))

            pheromone.update()
            coverage_progress.append(np.count_nonzero(visit_map) / visit_map.size)

        # Metrics
        coverage = np.count_nonzero(visit_map) / visit_map.size
        total_visits = np.sum(visit_map)
        unique_visits = np.count_nonzero(visit_map)
        overlap = (total_visits - unique_visits) / total_visits if total_visits > 0 else 0.0

        moves_per_drone = [max(0, len(d.path) - 1) for d in drones if d.active]
        energy = (np.mean(moves_per_drone) / timesteps) if moves_per_drone else 0.0

        explored_row_ratio = np.count_nonzero(np.sum(visit_map, axis=1)) / grid_size[0]
        explored_col_ratio = np.count_nonzero(np.sum(visit_map, axis=0)) / grid_size[1]

        visited = (visit_map > 0)
        mean_phero_visited = float(pheromone.map[visited].mean()) if visited.any() else 0.0

        late_exploration_bonus = coverage_progress[-1] - coverage_progress[int(0.5 * timesteps)]

        fmin, fmax = field.min(), field.max()
        health_norm = (field - fmin) / (fmax - fmin + 1e-9)
        stress = 1.0 - health_norm

        stress_covered = float((stress * visited).sum() / (stress.sum() + 1e-9))
        q = 0.90
        thresh = np.quantile(stress, q)
        hotspots = (stress >= thresh)
        hotspot_recall = float((visited & hotspots).sum() / (hotspots.sum() + 1e-9))
        coverage_auc = float(np.mean(coverage_progress))

        fitness = (
            30.0 * coverage
          - 40.0 * overlap
          - 0.5  * energy
          + 1.5  * explored_row_ratio
          + 1.5  * explored_col_ratio
          + 1.5  * late_exploration_bonus
          - 2.0  * (mean_phero_visited / (1.0 + pheromone.map.mean()))
          + 8.0  * np.sqrt(explored_row_ratio * explored_col_ratio)
          + 30.0 * stress_covered
          + 10.0 * hotspot_recall
          + 10.0 * coverage_auc
        )
        fitnesses.append(fitness)

    return np.mean(fitnesses)
