"""Learning curves of the convergence probe and a steps-to-converge estimate.

Reads the training ``episodes.csv`` of every run under an experiment
directory, bins episodes by environment step and reports, per K, the first
step at which the training-pool solve rate reaches the target in every seed.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BIN = 50_000              # environment steps per bin
TARGET_SOLVE = 0.90
# "flat": solve rate changes by less than this over the next FLAT_SPAN steps
FLAT_DELTA = 0.02
FLAT_SPAN = 250_000
PANELS = [
    ("solve_rate", "training-pool solve rate"),
    ("ep_len", "episode length"),
    ("cost", "cost per episode"),
]


def binned_curves(exp_dir):
    frames = []
    for done_path in sorted(Path(exp_dir).glob("*/done.json")):
        meta = json.loads(done_path.read_text())
        eps = pd.read_csv(done_path.parent / "episodes.csv")
        eps["step_bin"] = (eps["global_step"] // BIN + 1) * BIN
        curve = eps.groupby("step_bin").agg(
            solve_rate=("solved", "mean"), ep_len=("ep_len", "mean"),
            cost=("cost", "mean"), cost_c1=("cost_c1", "mean"),
            cost_c2=("cost_c2", "mean"), episodes=("solved", "size"),
        ).reset_index()
        curve.insert(0, "seed", meta["seed"])
        curve.insert(0, "k", meta["k"])
        curve.insert(0, "num_hosts", meta["num_hosts"])
        curve.insert(0, "method", meta["method"])
        frames.append(curve)
    return pd.concat(frames, ignore_index=True)


def steps_to_converge(curves):
    rows = []
    for (k, seed), curve in curves.groupby(["k", "seed"]):
        curve = curve.sort_values("step_bin").reset_index(drop=True)
        reached = curve[curve["solve_rate"] >= TARGET_SOLVE]
        flat_step = None
        for _, row in curve.iterrows():
            later = curve[(curve["step_bin"] > row["step_bin"])
                          & (curve["step_bin"] <= row["step_bin"] + FLAT_SPAN)]
            if (len(later) == FLAT_SPAN // BIN
                    and (later["solve_rate"] - row["solve_rate"]).abs().max()
                    < FLAT_DELTA):
                flat_step = int(row["step_bin"])
                break
        last = curve.tail(4)      # last 200k steps
        rows.append(dict(
            k=k, seed=seed,
            first_step_solve_ge_90=(int(reached["step_bin"].iloc[0])
                                    if len(reached) else ""),
            first_step_flat="" if flat_step is None else flat_step,
            final_solve_rate=round(last["solve_rate"].mean(), 4),
            final_ep_len=round(last["ep_len"].mean(), 2),
            final_cost=round(last["cost"].mean(), 3),
            total_steps=int(curve["step_bin"].max()),
        ))
    return pd.DataFrame(rows)


def run(exp_dir, out_dir=None):
    exp_dir = Path(exp_dir)
    out_dir = Path(out_dir) if out_dir else exp_dir
    curves = binned_curves(exp_dir)
    curves.to_csv(out_dir / "curve_bins.csv", index=False)
    converge = steps_to_converge(curves)
    converge.to_csv(out_dir / "steps_to_converge.csv", index=False)

    ks = sorted(curves["k"].unique())
    fig, axes = plt.subplots(len(PANELS), len(ks), sharex=True,
                             figsize=(5.2 * len(ks), 8.5), squeeze=False)
    for col, k in enumerate(ks):
        for row, (metric, label) in enumerate(PANELS):
            ax = axes[row][col]
            for seed, curve in curves[curves["k"] == k].groupby("seed"):
                ax.plot(curve["step_bin"] / 1e6, curve[metric],
                        label=f"seed {seed}")
            if metric == "solve_rate":
                ax.axhline(TARGET_SOLVE, color="gray", linestyle="--",
                           linewidth=1)
                ax.set_ylim(0, 1.02)
                ax.set_title(f"M1 (reward-only PPO), K={k}")
                ax.legend(loc="lower right", fontsize=8)
            if metric == "ep_len":
                ax.set_yscale("log")
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
        axes[-1][col].set_xlabel(f"environment steps (millions), {BIN // 1000}k bins")
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curves.png", dpi=150)

    with pd.option_context("display.width", 200):
        print(converge.to_string(index=False))
    print(f"wrote curve_bins.csv, steps_to_converge.csv, learning_curves.png in {out_dir}")
