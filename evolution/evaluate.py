import numpy as np
from agents.neural_controller import NeuralController
from agents.drone import Drone
from environment.field import generate_field
from environment.pheromone_map import PheromoneMap

def evaluate_controller(weights, grid_size=(20, 20), num_drones=5, timesteps=30, seeds=[42, 17, 73]):
    """
    Evaluate one neural controller's performance over multiple random seeds.
    Returns the average fitness across runs.
    """
    fitnesses = []

    for seed in seeds:
        np.random.seed(seed)

        field = generate_field(*grid_size, seed=seed)
        pheromone = PheromoneMap(grid_size, decay_rate=0.01)
        visit_map = np.zeros(grid_size, dtype=int)

        controller = NeuralController(weights=weights)

        # Drones
        drones = []
        for _ in range(num_drones):
            x = np.random.randint(0, grid_size[0])
            y = np.random.randint(0, grid_size[1])
            drone = Drone(x, y, grid_size, controller=controller, failure_prob=0.0)
            drone.visit_map_ref = visit_map
            drones.append(drone)

        # Simulation loop
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

        # Metrics
        coverage = np.count_nonzero(visit_map) / visit_map.size * 100
        total_visits = np.sum(visit_map)
        unique_visits = np.count_nonzero(visit_map)
        overlap = (total_visits - unique_visits) / total_visits * 100 if total_visits > 0 else 0
        energy = np.mean([len(d.path) - 1 for d in drones if d.active])

        w1, w2, w3 = 2.0, 1.0, 0.05
        fitness = w1 * coverage - w2 * overlap - w3 * energy
        explored_rows = np.count_nonzero(np.sum(visit_map, axis=1))
        coverage_reward = explored_rows / grid_size[0] * 100
        fitness += 0.5 * coverage_reward
        fitnesses.append(fitness)

    return np.mean(fitnesses)
