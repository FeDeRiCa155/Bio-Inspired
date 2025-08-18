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
    mutation_rate=0.3,
    mutation_strength=0.4,
    seed=1
):
    np.random.seed(seed)

    # Initialize random population
    sample_net = NeuralController()
    n_params = sample_net.n_params
    population = [np.random.uniform(-1, 1, n_params) for _ in range(population_size)]

    history = []
    best_so_far = -np.inf
    stagnation_counter = 0
    max_stagnation = 3
    restart_counter = 0

    for gen in range(generations):

        # Evaluate population
        # fitness_scores = [evaluate_controller(ind) for ind in population]
        seeds_train = [1, 15, 42, 38, 95, 63, 100, 7, 37]
        lambda_l2 = 1e-3
        fitness_scores = [evaluate_controller(ind, grid_size=(25, 25), seeds=seeds_train) - lambda_l2 * float(np.dot(ind, ind))
                          for ind in population]

        # Sort by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]

        best_fitness = fitness_scores[0]
        mean_fitness = np.mean(fitness_scores)
        std_fitness = np.std(fitness_scores)
        print(f"[Gen {gen}] Best: {best_fitness:.2f} | Mean: {mean_fitness:.2f} | Std: {std_fitness:.2f}")
        history.append((best_fitness, mean_fitness))

        # Save improvement
        if best_fitness > best_so_far + 1e-6:
            best_so_far = best_fitness
            stagnation_counter = 0
            np.save("best_weights_25_5.npy", population[0])
            print(f"[Gen {gen}] New best: {best_fitness:.2f} (saved)")
        else:
            stagnation_counter += 1

        # Select elites
        num_elite = max(1, int(population_size * elite_fraction))
        elites = population[:num_elite]

        # Adaptive mutation rate
        current_mut_rate = max(0.1, mutation_rate * (0.97 ** gen))

        # Reproduce new population
        new_population = elites.copy()
        while len(new_population) < population_size:
            p1 = tournament_selection(population, fitness_scores, k=3)
            p2 = tournament_selection(population, fitness_scores, k=3)
            child = crossover(p1, p2)
            child = mutate(child, current_mut_rate, mutation_strength)
            new_population.append(child)

        # Inject some random individuals for diversity
        num_random = max(1, population_size // 10)
        for _ in range(num_random):
            new_population[np.random.randint(num_elite, population_size)] = np.random.uniform(-1, 1, n_params)

        # Handle stagnation
        if stagnation_counter >= max_stagnation:
            print(f"Restarting some of the population due to stagnation")
            restart_counter += 1
            restart_pop = int(0.7 * population_size)
            for i in range(population_size - restart_pop, population_size):
                new_population[i] = np.random.uniform(-1, 1, n_params)
            if restart_counter % 5 == 0:
                print("Elite replaced")
                new_population[0] = np.random.uniform(-1, 1, n_params)
                restart_counter = 0
            stagnation_counter = 0

        population = new_population

    return population[0], history


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

