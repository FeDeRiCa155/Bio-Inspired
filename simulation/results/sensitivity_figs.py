import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

METRICS_PRIORITY = [
    ("coverage", "Coverage [%]"),
    ("overlap", "Overlap [%]"),
    ("energy", "Energy [mean moves / drone]"),
    ("explored_row_ratio", "Explored row ratio [-]"),
    ("explore_col_ratio", "Explored column ratio [-]"),
]
LINEWIDTH = 2.0
MARKERSIZE = 6

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def has_columns(df, cols):
    return all(c in df.columns for c in cols)

def plot_metric_vs_drones_by_failprob(df, metric, metric_label, outdir):
    """
    For each failure_prob: plot metric vs. num_drones.
    Produces one figure per failure_prob.
    """
    grouped = df.groupby("failure_prob")
    for fail_p, sub in grouped:
        if metric not in sub.columns or f"{metric}_std" not in sub.columns:
            continue
        sub = sub.sort_values(["num_drones", "method"])

        fig, ax = plt.subplots(figsize=(6, 4.2))
        for method, s2 in sub.groupby("method"):
            x = s2["num_drones"].values
            y = s2[metric].values
            yerr = s2[f"{metric}_std"].values
            ax.errorbar(x, y, yerr=yerr, linewidth=LINEWIDTH, marker='o', markersize=MARKERSIZE, label=method)

        ax.set_xlabel("Number of drones")
        ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label} vs. drones (failure prob = {fail_p:.3f})")
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(title="method")
        fname = f"{metric}_vs_drones_fail_{str(fail_p).replace('.', 'p')}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, fname), dpi=200)
        plt.close(fig)

def plot_metric_vs_failprob_by_drones(df, metric, metric_label, outdir):
    """
    For each num_drones: plot metric vs. failure_prob.
    Produces one figure per num_drones.
    """
    grouped = df.groupby("num_drones")
    for n, sub in grouped:
        if metric not in sub.columns or f"{metric}_std" not in sub.columns:
            continue
        sub = sub.sort_values(["failure_prob", "method"])

        fig, ax = plt.subplots(figsize=(6, 4.2))
        for method, s2 in sub.groupby("method"):
            x = s2["failure_prob"].values
            y = s2[metric].values
            yerr = s2[f"{metric}_std"].values
            ax.errorbar(x, y, yerr=yerr, linewidth=LINEWIDTH, marker='o', markersize=MARKERSIZE, label=method)

        ax.set_xlabel("Failure probability")
        ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label} vs. failure prob (drones = {n})")
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(title="method")
        fname = f"{metric}_vs_failprob_drones_{n}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, fname), dpi=200)
        plt.close(fig)

def main():
    # if len(sys.argv) < 3:
    #     print("Usage: python sensitivity_figs.py <path/to/sensitivity_agg.csv> <outdir>")
    #     sys.exit(1)

    default_csv = os.path.join("sensitivity", "20250820_152118", "sensitivity_agg.csv")

    default_out = ensure_dir("figs")

    if len(sys.argv) < 3:
        print(f"Running in default mode with CSV={default_csv} to {default_out}")
        agg_csv = default_csv
        outdir = default_out
    else:
        agg_csv = sys.argv[1]
        outdir = ensure_dir(sys.argv[2])

    df = pd.read_csv(agg_csv)

    required = ["method", "num_drones", "failure_prob"]
    if not has_columns(df, required):
        raise ValueError(f"CSV must contain columns: {required}. Found: {df.columns.tolist()}")

    # Normalize types
    df["num_drones"] = df["num_drones"].astype(int)
    df["failure_prob"] = df["failure_prob"].astype(float)

    present_metrics = [(m, lab) for (m, lab) in METRICS_PRIORITY if m in df.columns and f"{m}_std" in df.columns]
    if not present_metrics:
        raise ValueError("No known metrics found (with *_std). Check your CSV columns.")

    # for each metric, (1) vs drones per failprob, (2) vs failprob per drones
    for metric, metric_label in present_metrics:
        plot_metric_vs_drones_by_failprob(df, metric, metric_label, outdir)
        plot_metric_vs_failprob_by_drones(df, metric, metric_label, outdir)

    print(f"Figures to: {outdir}")

if __name__ == "__main__":
    main()