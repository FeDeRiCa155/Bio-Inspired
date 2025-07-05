from simulation.loop import run_simulation
from visual.plotter import (
    plot_field_with_paths,
    plot_pheromone_map,
    plot_visit_heatmap
)
from simulation.metrics import (
    compute_coverage,
    compute_overlap,
    compute_energy,
    compute_failures
)

field, pheromone, drones, visit_map = run_simulation()

coverage = compute_coverage(visit_map)
overlap = compute_overlap(visit_map)
mean_energy, std_energy = compute_energy(drones)
failures = compute_failures(drones)

print(f"Coverage: {coverage:.2f}%")
print(f"Overlap: {overlap:.2f}%")
print(f"Energy per drone (mean ± std): {mean_energy:.2f} ± {std_energy:.2f}")
print(f"Failed drones: {failures}/{len(drones)}")

plot_field_with_paths(field, drones)
plot_pheromone_map(pheromone.map)
plot_visit_heatmap(visit_map)

# test evolution
from evolution.evolve import evolve
best_weights, history = evolve(generations=30, population_size=30)

