import numpy as np
from agents.neural_controller import NeuralController
from agents.drone import Drone
from environment.pheromone_map import PheromoneMap
from environment.field import generate_field

grid_size = (20, 20)

# Load evolved controller weights
best_weights = np.load("best_weights.npy")
controller = NeuralController(weights=best_weights)

pheromone = PheromoneMap(grid_size, decay_rate=0.05)
visit_map = np.zeros(grid_size, dtype=int)

drone = Drone(x=0, y=0, field_shape=grid_size, controller=controller, visit_map=visit_map)
field = generate_field(*grid_size, seed=42)

for step in range(10):
    drone.decide_and_move(
        field=field,
        pheromone_map=pheromone.map,
        occupied_positions=set()
    )
    pheromone.deposit(drone.x, drone.y, amount=1.0)
    pheromone.update()
    print(f"Step {step+1}: Position=({drone.x},{drone.y})")

print("\nFinal pheromone map:")
print(np.round(pheromone.map, 2))

