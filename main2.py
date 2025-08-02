import numpy as np
import matplotlib.pyplot as plt
import csv

from agents.neural_controller import NeuralController
from evolution.evolve import evolve
from simulation.loop import run_simulation
from simulation.metrics import compute_all_metrics
from visual.plotter import (
    plot_field_with_paths,
    plot_pheromone_map,
    plot_visit_heatmap
)

# train controller
best_weights, history = evolve(generations=30, population_size=30)
controller = NeuralController(weights=best_weights)

# simulation loop
field, pheromone, drones, visit_map = run_simulation(
    grid_size=(20, 20),
    num_drones=5,
    timesteps=300,
    failure_prob=0.0,
    seed=42
)

for drone in drones:
    drone.controller = controller
    drone.visit_map_ref = visit_map

# post-training simulation
for _ in range(100):
    active_drones = [d for d in drones if d.active]
    occupied = {(d.x, d.y) for d in active_drones}
    all_pos = [(d.x, d.y) for d in active_drones]
    for drone in drones:
        drone.maybe_fail()
        drone.decide_and_move(field, pheromone.map, occupied, all_pos)
        if drone.active:
            visit_map[drone.x, drone.y] += 1
        drone.deposit_pheromone(pheromone)
    pheromone.update()

# metrics
metrics = compute_all_metrics(visit_map, drones)

print("\n-----Simulation Metrics")
for k, v in metrics.items():
    print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")

# Plots
plot_field_with_paths(field, drones)
plot_visit_heatmap(visit_map)
plot_pheromone_map(pheromone.map)

best, mean = zip(*history)
plt.plot(best, label="Best")
plt.plot(mean, label="Mean")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Evolution Progress")
plt.legend()
plt.grid(True)
plt.show()

