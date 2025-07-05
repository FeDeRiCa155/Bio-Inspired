import numpy as np
from agents.drone import Drone
from environment.field import generate_field
from environment.pheromone_map import PheromoneMap
from agents.neural_controller import NeuralController

def run_simulation(
    grid_size=(30, 30),
    num_drones=10,
    timesteps=100,
    pheromone_decay=0.01,
    seed=42,
    failure_prob=0.01
):
    """
    Runs the swarm simulation.

    Returns:
        field (np.ndarray): Crop health map.
        pheromone (PheromoneMap): Final pheromone state.
        drones (list): List of Drone objects.
    """
    np.random.seed(seed)

    field = generate_field(*grid_size, seed=seed)
    visit_map = np.zeros(grid_size, dtype=int)
    pheromone = PheromoneMap(grid_size, decay_rate=pheromone_decay)

    drones = []
    for _ in range(num_drones):
        x = np.random.randint(0, grid_size[0])
        y = np.random.randint(0, grid_size[1])
        # drones.append(Drone(x, y, grid_size, failure_prob=failure_prob))
        controller = NeuralController()  # or None for rule-based
        drone = Drone(x, y, grid_size, controller=controller)
        drone.visit_map_ref = visit_map  # Attach visit map reference
        drones.append(drone)

    for t in range(timesteps):
        active_drones = [d for d in drones if d.active]
        occupied_positions = {(d.x, d.y) for d in active_drones}
        all_positions = [(d.x, d.y) for d in active_drones]

        for drone in drones:
            drone.maybe_fail()
            drone.decide_and_move(field, pheromone.map, occupied_positions, all_positions=all_positions)
            if drone.active:
                visit_map[drone.x, drone.y] += 1
            drone.deposit_pheromone(pheromone)

        pheromone.update()
        num_failed = sum(not d.active for d in drones)
        print(f"Simulation complete. {num_failed}/{num_drones} drones failed.")

    return field, pheromone, drones, visit_map
