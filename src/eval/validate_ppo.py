"""Check the hand-written PPO against SB3 PPO on NASim ``tiny``.

Same hyperparameters, same seeds, both learning curves in one figure.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.agents.ppo import PPOConfig, train
from src.envs.compat import make_benchmark
from src.phase0 import sb3_check

WINDOW = 100          # episodes in the rolling mean
EVAL_EPISODES = 20


def _eval_sampled(agent, max_steps=1000):
    env = make_benchmark("tiny")
    lengths, solved = [], []
    for _ in range(EVAL_EPISODES):
        obs, _ = env.reset()
        for step in range(1, max_steps + 1):
            obs, _, terminated, truncated, _ = env.step(agent.act(obs))
            if terminated or truncated:
                break
        lengths.append(step)
        solved.append(int(terminated))
    return float(np.mean(solved)), float(np.mean(lengths))


def run(total_steps, seeds, out_dir):
    out_dir = Path(out_dir)
    rows = []
    for seed in seeds:
        run_dir = out_dir / f"own_tiny_po_seed{seed}"
        if not (run_dir / "train_summary.json").exists():
            cfg = PPOConfig(total_steps=total_steps, num_envs=4, seed=seed)
            agent, summary = train(
                [lambda: make_benchmark("tiny")] * cfg.num_envs, cfg, run_dir
            )
            solve_rate, mean_len = _eval_sampled(agent)
            pd.DataFrame([dict(
                impl="own", seed=seed, train_steps=summary["train_steps"],
                train_wall_s=summary["train_wall_s"],
                steps_per_sec=summary["train_steps_per_sec"],
                last100_train_ep_len=summary["last100_ep_len"],
                eval_sampled_solve_rate=solve_rate,
                eval_sampled_ep_len=mean_len,
            )]).to_csv(run_dir / "result.csv", index=False)
        rows.append(pd.read_csv(run_dir / "result.csv").iloc[0].to_dict())

        sb3 = sb3_check.run_one("tiny", False, seed, total_steps, out_dir)
        sb3_eps = pd.read_csv(out_dir / f"sb3_tiny_po_seed{seed}" / "episodes.csv")
        rows.append(dict(
            impl="sb3", seed=seed, train_steps=sb3["train_steps"],
            train_wall_s=sb3["train_wall_s"],
            steps_per_sec=sb3["train_steps_per_sec"],
            last100_train_ep_len=float(sb3_eps["ep_len"].tail(100).mean()),
            eval_sampled_solve_rate=sb3["eval_stoch_solve_rate"],
            eval_sampled_ep_len="",
        ))

    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(7, 4))
    for impl, color in (("own", "tab:blue"), ("sb3", "tab:orange")):
        for i, seed in enumerate(seeds):
            eps = pd.read_csv(out_dir / f"{impl}_tiny_po_seed{seed}" / "episodes.csv")
            ax.plot(eps["global_step"], eps["ep_len"].rolling(WINDOW).mean(),
                    color=color, alpha=0.8,
                    label=("hand-written PPO" if impl == "own" else "SB3 PPO")
                    if i == 0 else None)
    ax.set_yscale("log")
    ax.set_xlabel("environment steps")
    ax.set_ylabel(f"episode length (rolling mean of {WINDOW})")
    ax.set_title(f"NASim tiny, partially observable, {len(seeds)} seeds each")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "ppo_vs_sb3_tiny.png", dpi=150)
    for row in rows:
        print(row)
    print(f"wrote {out_dir / 'summary.csv'} and ppo_vs_sb3_tiny.png")
