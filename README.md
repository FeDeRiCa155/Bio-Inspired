# Bio-Inspired REPOSITORY
This repository contains the code for the AE4350 assignment on bio-inspired intelligence for aerospace applications. 



**Project Overview**

The project explores swarm coordination strategies for agricultural crop monitoring using autonomous UAVs.

1. GOAL: Investigate bio-inspired methods for distributed UAV coverage and detection of unhealthy crop zones.

2. Approaches

- Rule-based heuristic controller using stigmergic interactions (pheromone maps and a desirability function).
- Neuro-evolutionary controller where agents evolve neural network policies using a genetic algorithm.
- Comparison: Both methods are evaluated on coverage, overlap, energy efficiency, and robustness under drone failures.

3. Implementation
- The environment is a 2D grid representing a crop field with healthy, moderate, and sick zones.
- Agents (drones) operate with local sensing, pheromone-based stigmergy, and decentralized decision-making.
- Neuro-evolution uses a genetic algorithm to evolve neural network controllers over multiple generations.
- Simulation results include trajectory plots, coverage maps, and sensitivity analyses.

4. Results
- The heuristic method achieves structured dispersion, high coverage, and low overlap, scaling predictably with swarm size.
- The neuro-evolutionary controller shows adaptability and crop-health prioritization but suffers from band-like trajectories and repeated revisits.
- Sensitivity analysis highlights differences in robustness under varying swarm sizes and drone failure probabilities.

5. Future Work
- Hybrid methods that combine heuristic stigmergic rules with adaptive neuro-evolutionary controllers.
- Higher-fidelity simulations including UAV dynamics, wind disturbances, and sensor noise.
- Pathways toward real-world deployment in agricultural monitoring scenarios.
