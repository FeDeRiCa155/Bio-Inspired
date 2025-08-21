import numpy as np
from agents.neural_controller import NeuralController
from evolution.evaluate import evaluate_controller
# from evolution.evolve import evolve

def crossover(parent1, parent2):
    """Uniform crossover."""
    mask = np.random.rand(len(parent1)) < 0.5
    child = np.where(mask, parent1, parent2)
    return child


def mutate(weights, mutation_rate, mutation_strength):
    """Gaussian mutation."""
    w = weights.copy()
    mask = np.random.rand(w.size) < mutation_rate
    w[mask] += np.random.randn(mask.sum()) * mutation_strength
    return w


def tournament_selection(population, fitness_scores, k=3):
    """Tournament selection."""
    indices = np.random.choice(len(population), k, replace=False)
    best_idx = max(indices, key=lambda idx: fitness_scores[idx])
    return population[best_idx]


def _eval_single(weights, s):
    """Call evaluate_controller; try passing seed, else set global seed."""
    try:
        return evaluate_controller(weights, seed=s)
    except TypeError:
        np.random.seed(int(s))
        return evaluate_controller(weights)


def eval_fitness_mean(weights, seeds):
    """Average fitness across seeds."""
    scores = np.array([_eval_single(weights, s) for s in seeds], dtype=float)
    return float(scores.mean())


def evolve(
    generations=20,
    population_size=30,
    elite_fraction=0.1,
    mutation_rate=0.4,
    mutation_strength=0.4,
    seed=5
):
    np.random.seed(seed)

    # Initialize random population
    sample_net = NeuralController()
    n_params = sample_net.n_params
    population = [np.random.uniform(-1, 1, n_params) for _ in range(population_size)]

    history = []
    best_so_far = -np.inf
    best_weights_ever = population[0].copy()
    stagnation_counter = 0
    max_stagnation = 3
    restart_counter = 0

    lambda_l2 = 1e-3 / max(1, n_params)

    base_seeds = [1, 15, 42, 58, 95, 63, 100, 7, 37, 24]
    seeds_train = list(base_seeds)

    current_mut_rate = mutation_rate
    current_mut_strength = float(mutation_strength)

    for gen in range(generations):

        # Light seed rotation every 5 generations to avoid overfitting to a fixed set
        if gen > 0 and gen % 5 == 0:
            np.random.shuffle(base_seeds)
            seeds_train = base_seeds[:8] + list(np.random.choice(base_seeds, 2, replace=False))

        # Evaluate population (L2-regularized)
        fitness_scores = []
        for ind in population:
            fit = evaluate_controller(ind, grid_size=(25, 25), seeds=seeds_train)
            fit -= lambda_l2 * float(np.dot(ind, ind))
            fitness_scores.append(fit)

        # Sort by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]

        best_fitness = fitness_scores[0]
        mean_fitness = float(np.mean(fitness_scores))
        std_fitness = float(np.std(fitness_scores))
        print(f"[Gen {gen}] Best: {best_fitness:.2f} | Mean: {mean_fitness:.2f} | Std: {std_fitness:.2f}")
        history.append((best_fitness, mean_fitness))

        # Track improvement & adapt mutation *strength*
        if best_fitness > best_so_far + 1e-6:
            best_so_far = best_fitness
            best_weights_ever = population[0].copy()
            stagnation_counter = 0
            current_mut_strength = max(0.15, current_mut_strength * 0.9)
            np.save("best_weights_25.npy", best_weights_ever)
            print(f"[Gen {gen}] New best: {best_fitness:.2f} (saved)")
        else:
            stagnation_counter += 1
            current_mut_strength = min(0.8, current_mut_strength * 1.25)

        # Select elites (slightly lower pressure helps diversity)
        num_elite = max(1, int(population_size * max(0.05, elite_fraction)))
        elites = population[:num_elite]

        # Reproduce new population
        new_population = elites.copy()
        while len(new_population) < population_size:
            p1 = tournament_selection(population, fitness_scores, k=3)
            p2 = tournament_selection(population, fitness_scores, k=3)
            child = crossover(p1, p2)
            child = mutate(child, current_mut_rate, current_mut_strength)
            new_population.append(child)

        # Light immigrants every gen for diversity
        num_random = max(1, population_size // 10)
        for _ in range(num_random):
            idx = np.random.randint(num_elite, population_size)
            new_population[idx] = np.random.uniform(-1, 1, n_params)

        # Heavier immigration on stagnation
        if stagnation_counter >= max_stagnation:
            print("Injecting random immigrants due to stagnation")
            n_imm = max(1, int(0.25 * population_size))
            start = num_elite
            end = min(population_size, start + n_imm)
            for j in range(start, end):
                new_population[j] = np.random.uniform(-1, 1, n_params)
            stagnation_counter = 0
            # after heavy injection, slightly reduce strength (stabilize)
            current_mut_strength = max(0.3, current_mut_strength * 0.8)
            restart_counter += 1
            if restart_counter % 5 == 0:
                print("Elite replaced to avoid lock-in")
                new_population[0] = np.random.uniform(-1, 1, n_params)
                restart_counter = 0

        population = new_population

    return best_weights_ever, history



if __name__ == "__main__":
    best_weights, history = evolve(
        generations=30,
        population_size=30,
        elite_fraction=0.15,
        mutation_rate=0.5,
        mutation_strength=0.5,
        seed=15
    )
    print("Best weights saved")

