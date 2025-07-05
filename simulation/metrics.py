import numpy as np

def compute_coverage(visit_map):
    total_cells = visit_map.size
    visited_cells = np.count_nonzero(visit_map)
    return visited_cells / total_cells * 100

def compute_overlap(visit_map):
    total_visits = np.sum(visit_map)
    single_visits = np.count_nonzero(visit_map == 1)
    overlap_visits = total_visits - single_visits
    return overlap_visits / total_visits * 100 if total_visits > 0 else 0

def compute_energy(drones):
    energies = [len(d.path) - 1 for d in drones if d.active]
    return np.mean(energies), np.std(energies)

def compute_failures(drones):
    return sum(not d.active for d in drones)
