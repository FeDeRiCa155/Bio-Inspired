import argparse
import itertools
import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
from agents.neural_controller import NeuralController
from evolution.evolve import evolve
from simulation.loop import run_simulation, generate_start_positions
from simulation.metrics import compute_all_metrics


def run_one(grid_size, num_drones, timesteps, failure_prob, seed, method, controller):
    """
    Run a single simulation and return a flat dict of metrics + config.
    """
    start_positions = generate_start_positions(grid_size=grid_size, num_drones=num_drones)
    field, pheromone, drones, visit_map = run_simulation(
        grid_size=grid_size,
        num_drones=num_drones,
        timesteps=timesteps,
        failure_prob=failure_prob,
        seed=seed,
        controller=controller,
        start_positions=start_positions
    )
    metrics = compute_all_metrics(visit_map, drones)

    flat = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float, np.floating, np.integer)):
            flat[k] = float(v)
        elif hasattr(v, "shape") and v.shape == ():
            flat[k] = float(v)

    flat.update({
        "method": method,
        "num_drones": num_drones,
        "timesteps": timesteps,
        "failure_prob": failure_prob,
        "seed": seed,
        "grid_h": grid_size[0],
        "grid_w": grid_size[1],
    })
    return flat


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def main():
    parser = argparse.ArgumentParser(description="Sensitivity analysis for swarm simulation.")
    parser.add_argument("--grid", type=int, nargs=2, default=[20, 20], help="Grid size H W")
    parser.add_argument("--timesteps", type=int, default=100, help="Simulation steps")
    parser.add_argument("--methods", type=str, nargs="+", default=["neuro", "heuristic"],
                        choices=["neuro", "heuristic"], help="Control methods to compare")
    parser.add_argument("--drones", type=int, nargs="+", default=[1, 3, 5, 10], help="Number of drones to test")
    parser.add_argument("--fail_probs", type=float, nargs="+", default=[0.0, 0.005, 0.01, 0.05],
                        help="Failure probabilities to test")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="Seeds for repetition")
    parser.add_argument("--weights", type=str, default="best_weights.npy",
                        help="Path to weights for neuro controller (if not training)")
    parser.add_argument("--train", action="store_true",
                        help="If set, evolve a controller now instead of loading weights.")
    parser.add_argument("--generations", type=int, default=20, help="If --train, number of generations")
    parser.add_argument("--popsize", type=int, default=30, help="If --train, population size")
    parser.add_argument("--outdir", type=str, default="results/sensitivity", help="Output folder")
    args = parser.parse_args()

    grid_size = (args.grid[0], args.grid[1])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = ensure_dir(os.path.join(args.outdir, timestamp))

    neuro_weights = None
    if "neuro" in args.methods:
        if args.train:
            print(f"[INFO] Evolving controller: generations={args.generations}, pop={args.popsize}")
            neuro_weights, history = evolve(generations=args.generations, population_size=args.popsize)
            np.save(os.path.join(outdir, "best_weights_evolved.npy"), neuro_weights)
            # Optionally save evolution history
            hist_df = pd.DataFrame(history, columns=["best", "mean"])
            hist_df.to_csv(os.path.join(outdir, "evolution_history.csv"), index=False)
            print("[INFO] Saved evolved weights and history.")
        else:
            if not os.path.exists(args.weights):
                raise FileNotFoundError(f"Weights file not found: {args.weights}")
            neuro_weights = np.load(args.weights)

    combos = list(itertools.product(args.methods, args.drones, args.fail_probs, args.seeds))
    print(f"[INFO] Running {len(combos)} simulations "
          f"({len(args.methods)} methods × {len(args.drones)} drone counts × "
          f"{len(args.fail_probs)} fail probs × {len(args.seeds)} seeds)")

    t0 = time.time()
    rows = []
    for (method, num_drones, fail_p, seed) in combos:
        # Build controller per method
        controller = None
        if method == "neuro":
            controller = NeuralController(weights=neuro_weights)

        row = run_one(
            grid_size=grid_size,
            num_drones=num_drones,
            timesteps=args.timesteps,
            failure_prob=fail_p,
            seed=seed,
            method=method,
            controller=controller
        )
        rows.append(row)

        if len(rows) % 5 == 0:
            elapsed = time.time() - t0
            print(f"[{len(rows)}/{len(combos)}] elapsed {elapsed:.1f}s")

    runs_df = pd.DataFrame(rows)
    runs_path = os.path.join(outdir, "sensitivity_runs.csv")
    runs_df.to_csv(runs_path, index=False)

    group_cols = ["method", "num_drones", "failure_prob"]
    numeric_cols = [c for c in runs_df.select_dtypes(include=[np.number]).columns
                    if c not in group_cols + ["seed"]]
    runs_df["failure_prob"] = runs_df["failure_prob"].astype(float).round(4)
    agg_mean = runs_df.groupby(group_cols)[numeric_cols].mean().reset_index()
    agg_std  = runs_df.groupby(group_cols)[numeric_cols].std(ddof=1).reset_index()
    std_renamed = {c: f"{c}_std" for c in numeric_cols}
    agg_std = agg_std.rename(columns=std_renamed)
    agg_df = pd.merge(agg_mean, agg_std, on=group_cols, how="left")

    agg_path = os.path.join(outdir, "sensitivity_agg.csv")
    agg_df.to_csv(agg_path, index=False)

    key_metrics = [m for m in ["coverage", "overlap", "energy", "explored_row_ratio", "explore_col_ratio"]
                   if m in runs_df.columns]
    if key_metrics:
        print("\n=== Aggregated (mean ± std) ===")
        for _, r in agg_df.sort_values(group_cols).iterrows():
            label = f"method={r['method']}, drones={int(r['num_drones'])}, fail={r['failure_prob']:.3f}"
            parts = []
            for m in key_metrics:
                mu = r.get(m, np.nan)
                sd = r.get(f"{m}_std", np.nan)
                parts.append(f"{m}={mu:.3f}±{sd:.3f}")
            print(label, " | ", "  ".join(parts))
    else:
        print("\n[WARN] Key metric names not found in compute_all_metrics(). Check columns in runs CSV.")

    print(f"\n[OK] Wrote:\n  - per-run: {runs_path}\n  - aggregated: {agg_path}")


if __name__ == "__main__":
    main()
