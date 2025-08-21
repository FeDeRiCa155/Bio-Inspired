import numpy as np
import matplotlib.pyplot as plt
import csv

from agents.neural_controller import NeuralController
from simulation.run_GA import evolve
from simulation.loop import run_simulation, generate_start_positions
from simulation.metrics import compute_all_metrics, compute_fitness
from evolution.evaluate import evaluate_controller
from visual.plotter import (
    plot_field_with_paths,
    plot_pheromone_map,
    plot_visit_heatmap
)

# train controller
best_weights, history = evolve(generations=20, population_size=50)
# best_weights = np.load("best_weights_25.npy") # best_weights_25_try2.npy
controller = NeuralController(weights=best_weights)

num_drones = 2
# simulation
start_positions = generate_start_positions(grid_size=(25, 25), num_drones=num_drones)
field, pheromone, drones, visit_map = run_simulation(
    grid_size=(25, 25),
    num_drones=num_drones,
    timesteps=200,
    failure_prob=0.0,
    seed=69,
    controller=controller,
    start_positions=start_positions
)

# metrics
metrics = compute_all_metrics(visit_map, drones)

print("\n-----Simulation Metrics")
for k, v in metrics.items():
    print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")
# final_score = evaluate_controller(controller.get_weights(), timesteps=100, seeds=[999, 1001, 1005])
# print(f"Final Evaluation on Unseen Seeds: Avg Fitness = {final_score:.2f}")

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

