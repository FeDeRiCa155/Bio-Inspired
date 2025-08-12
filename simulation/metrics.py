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

def compute_explored_rows(visit_map):
    return np.count_nonzero(np.sum(visit_map, axis=1))

def compute_explored_cols(visit_map):
    return np.count_nonzero(np.sum(visit_map, axis=0))

def compute_explored_row_ratio(visit_map):
    return compute_explored_rows(visit_map) / visit_map.shape[0] * 100

def compute_explored_col_ratio(visit_map):
    return compute_explored_cols(visit_map) / visit_map.shape[0] * 100

def compute_fitness(coverage, overlap, energy, explored_row_ratio, explore_col_ratio,
                    w1=20.0, w2=10.0, w3=0.5, w4=2.0, w5=2.0, reg_strength=0.001):
    fitness = w1 * coverage - w2 * overlap - w3 * energy + w4 * explored_row_ratio + w5 * explore_col_ratio
    fitness -= reg_strength * (w1**2 + w2**2 + w3**2 + w4**2 + w5**2)
    return fitness

def compute_all_metrics(visit_map, drones):
    coverage = compute_coverage(visit_map)
    overlap = compute_overlap(visit_map)
    energy, std = compute_energy(drones)
    failures = compute_failures(drones)
    explored_row_ratio = compute_explored_row_ratio(visit_map)
    explored_col_ratio = compute_explored_col_ratio(visit_map)

    return {
        "coverage": coverage,
        "overlap": overlap,
        "energy_mean": energy,
        "energy_std": std,
        "failures": failures,
        "explored_row_ratio": explored_row_ratio,
        "explored_col_ratio": explored_col_ratio,
        "fitness": compute_fitness(coverage, overlap, energy, explored_row_ratio, explored_col_ratio)
    }

