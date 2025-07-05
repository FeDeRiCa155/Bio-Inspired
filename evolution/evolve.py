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

def evolve(
    generations=20,
    population_size=20,
    elite_fraction=0.2,
    mutation_rate=0.1,
    mutation_strength=0.3,
    seed=0
):
    np.random.seed(seed)

    # Random population
    sample_net = NeuralController()
    n_params = sample_net.n_params
    population = [np.random.uniform(-1, 1, n_params) for _ in range(population_size)]

    history = []

    for gen in range(generations):
        # Evaluation
        fitness_scores = [evaluate_controller(ind) for ind in population]

        # Sort by fitness (higher = better)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]

        best_fitness = fitness_scores[0]
        mean_fitness = np.mean(fitness_scores)
        print(f"[Gen {gen}] Best: {best_fitness:.2f} | Mean: {mean_fitness:.2f}")
        history.append((best_fitness, mean_fitness))

        # Select best (elites)
        num_elite = int(population_size * elite_fraction)
        elites = population[:num_elite]

        # Reproduction via mutation
        new_population = elites.copy()
        while len(new_population) < population_size:
            parent = elites[np.random.randint(num_elite)]
            child = mutate(parent.copy(), mutation_rate, mutation_strength)
            new_population.append(child)

        population = new_population

    # Return best controller
    best_weights = population[0]
    return best_weights, history
