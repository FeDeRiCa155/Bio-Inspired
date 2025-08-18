import numpy as np
from agents.neural_controller import NeuralController
from evolution.evaluate import evaluate_controller

def mutate(weights, mutation_rate=0.5, mutation_strength=0.5):
    """
    Add Gaussian noise to weights.
    """
    noise = np.random.randn(*weights.shape) * mutation_strength
    mask = np.random.rand(*weights.shape) < mutation_rate
    return weights + noise * mask

def crossover(parent1, parent2):
    """
    Crossover two parents.
    """
    mask = np.random.rand(*parent1.shape) < 0.5
    return np.where(mask, parent1, parent2)

def tournament_selection(pop, fitness, k=3):
    selected = np.random.choice(len(pop), k, replace=False)
    best_idx = selected[np.argmax([fitness[i] for i in selected])]
    return pop[best_idx]

def evolve(
    generations=30,
    population_size=30,
    elite_fraction=0.1,
    mutation_rate=0.5,
    mutation_strength=0.4,
    seed=1
):
    np.random.seed(seed)

    # Random population
    sample_net = NeuralController()
    n_params = sample_net.n_params
    population = [np.random.uniform(-1, 1, n_params) for _ in range(population_size)]

    history = []
    best_so_far = -np.inf
    stagnation_counter = 0
    max_stagnation = 3
    restart_counter = 0

    for gen in range(generations):
        # Evaluation
        fitness_scores = [evaluate_controller(ind) for ind in population]

        # Sort by fitness (higher = better)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]

        best_fitness = fitness_scores[0]
        mean_fitness = np.mean(fitness_scores)
        std_fitness = np.std(fitness_scores)
        print(f"[Gen {gen}] Best: {best_fitness:.2f} | Mean: {mean_fitness:.2f} | Std: {std_fitness:.2f}")
        history.append((best_fitness, mean_fitness))

        if best_fitness > best_so_far + 1e-6:
            best_so_far = best_fitness
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        # Select best (elites)
        num_elite = int(population_size * elite_fraction)
        elites = population[:num_elite]

        # Reproduction via mutation
        new_population = elites.copy()
        while len(new_population) < population_size:
            p1 = tournament_selection(population, fitness_scores, k=3)
            p2 = tournament_selection(population, fitness_scores, k=3)
            # p1 = elites[np.random.randint(num_elite)]
            # p2 = elites[np.random.randint(num_elite)]
            child = crossover(p1, p2)
            child = mutate(child, mutation_rate, mutation_strength)
            new_population.append(child)

        num_random = max(1, population_size // 10)
        worst_indices = sorted_indices[-num_random:]
        for idx in worst_indices:
            new_population.append(np.random.uniform(-1, 1, n_params))

        if stagnation_counter >= max_stagnation:
            print(f"Restarting some of the population due to stagnation")
            restart_counter += 1
            restart_pop = int(0.7 * population_size)
            for i in range(population_size - restart_pop, population_size):
                new_population[i] = np.random.uniform(-1, 1, n_params)
            if restart_counter % 3 == 0:
                print("Elite replaced")
                new_population[0] = np.random.uniform(-1, 1, n_params)
                restart_counter = 0
            stagnation_counter = 0

        population = new_population

    # Return best controller
    best_weights = population[0]
    return best_weights, history
