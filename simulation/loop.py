import numpy as np
from agents.drone import Drone
from environment.field import generate_field
from environment.pheromone_map import PheromoneMap
from agents.neural_controller import NeuralController

def generate_start_positions(grid_size, num_drones):
    """
    Evenly spread initial positions in the field to promote exploration.
    """
    positions = []
    rows = int(np.ceil(np.sqrt(num_drones)))
    cols = int(np.ceil(num_drones / rows))

    step_x = grid_size[0] // (rows + 1)
    step_y = grid_size[1] // (cols + 1)

    for i in range(rows):
        for j in range(cols):
            if len(positions) < num_drones:
                x = step_x * (i + 1)
                y = step_y * (j + 1)
                positions.append((x, y))

    return positions

def run_simulation(
    grid_size=(30, 30),
    num_drones=10,
    timesteps=300,
    pheromone_decay=0.05,
    seed=42,
    failure_prob=0.01,
    controller=None,
    start_positions=None
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

    # if controller is None:
    #     controller = NeuralController()

    if start_positions is None:
        start_positions = generate_start_positions(grid_size, num_drones)

    drones = []
    for (x, y) in start_positions:
        drone = Drone(x, y, grid_size, controller=controller, visit_map=visit_map, failure_prob=failure_prob)
        drones.append(drone)

    for t in range(timesteps):
        active = [d for d in drones if d.active]
        order = np.random.permutation(len(active))
        occupied_positions = {(d.x, d.y) for d in active}
        all_positions = [(d.x, d.y) for d in active]

        for k in order:
            d = active[k]
            d.maybe_fail()
            if not d.active:
                continue

            old = (d.x, d.y)
            occupied_positions.discard(old)

            d.decide_and_move(field, pheromone.map, occupied_positions, all_positions)

            moved = (d.x, d.y) != old
            if moved and d.active:
                visit_map[d.x, d.y] += 1
                d.deposit_pheromone(pheromone)

            occupied_positions.add((d.x, d.y))
            all_positions.append((d.x, d.y))

        pheromone.update()

    num_failed = sum(not d.active for d in drones)
    print(f"Simulation complete. {num_failed}/{num_drones} drones failed.")

    return field, pheromone, drones, visit_map
