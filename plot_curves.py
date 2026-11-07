import argparse
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_plots(results_dir: str, fixed_wait=None, maxpressure_wait=None):
    style = "seaborn-v0_8-whitegrid"
    plt.style.use(style if style in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    def plot_runs(pattern, label, color):
        files = glob.glob(os.path.join(results_dir, pattern))
        if not files:
            return
        runs = [pd.read_csv(f) for f in files]
        n = min(len(r) for r in runs)
        rewards = np.array([r["reward"].values[:n] for r in runs])
        waits = np.array([r["avg_waiting_time"].values[:n] for r in runs])
        ep = np.arange(1, n + 1)
        for ax, data in ((ax1, rewards), (ax2, waits)):
            m, s = data.mean(axis=0), data.std(axis=0)
            ax.plot(ep, m, label=label, color=color, linewidth=2)
            ax.fill_between(ep, m - s, m + s, color=color, alpha=0.15)

    plot_runs("independent_dqn_seed*.csv", "Independent DQN", "#1f77b4")
    plot_runs("shared_dqn_seed*.csv", "Shared-Parameter DQN", "#2ca02c")

    # Baseline lines only if you measured them yourself with this environment.
    if fixed_wait is not None:
        ax2.axhline(y=fixed_wait, color="#d62728", linestyle="--", label="Fixed-Time")
    if maxpressure_wait is not None:
        ax2.axhline(y=maxpressure_wait, color="#ff7f0e", linestyle=":", label="Max-Pressure")

    ax1.set_title("Episodic Cumulative Return", fontweight="bold")
    ax1.set_xlabel("Episode"); ax1.set_ylabel("Return"); ax1.legend(loc="lower right")
    ax2.set_title("Average Waiting Time (s)", fontweight="bold")
    ax2.set_xlabel("Episode"); ax2.set_ylabel("Waiting Time (s)"); ax2.legend(loc="upper right")

    plt.tight_layout()
    out = os.path.join(results_dir, "learning_curves.png")
    plt.savefig(out, dpi=300)
    print(f"Generated plot: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--fixed_wait", type=float, default=None)
    p.add_argument("--maxpressure_wait", type=float, default=None)
    a = p.parse_args()
    generate_plots(a.results, a.fixed_wait, a.maxpressure_wait)
