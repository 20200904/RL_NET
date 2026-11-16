"""Why did deterministic (argmax) evaluation score 0% on tiny in Phase 0?

Trains SB3 PPO on tiny, then compares argmax and sampled rollouts: solve
rate, episode length, how many distinct actions are used and how long the
longest run of one repeated action is.
"""
import csv
from itertools import groupby
from pathlib import Path

import numpy as np
import torch

from src.envs.compat import make_benchmark

EPISODES = 20
NUM_ENVS = 4


def _rollout(model, env, deterministic):
    obs, _ = env.reset()
    actions = []
    max_steps = env.unwrapped.scenario.step_limit or 1000
    solved = False
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        actions.append(int(action))
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            solved = bool(terminated)
            break
    return solved, actions


def _summarise(model, env, deterministic):
    solved, lengths, distinct, longest = [], [], [], []
    first_actions = None
    for _ in range(EPISODES):
        ok, actions = _rollout(model, env, deterministic)
        if first_actions is None:
            first_actions = actions[:25]
        solved.append(ok)
        lengths.append(len(actions))
        distinct.append(len(set(actions)))
        longest.append(max(len(list(g)) for _, g in groupby(actions)))
    return dict(
        solve_rate=float(np.mean(solved)),
        mean_len=float(np.mean(lengths)),
        mean_distinct_actions=float(np.mean(distinct)),
        mean_longest_repeat=float(np.mean(longest)),
        first_episode_actions=" ".join(map(str, first_actions)),
    )


def run(steps_list=(30_000, 200_000), out_dir=Path("results/phase0"), seed=0):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    torch.set_num_threads(1)
    rows = []
    for fully_obs in (True, False):
        for steps in steps_list:
            vec_env = DummyVecEnv([
                (lambda: make_benchmark("tiny", fully_obs=fully_obs))
                for _ in range(NUM_ENVS)
            ])
            model = PPO(
                "MlpPolicy", vec_env, seed=seed, device="cpu", verbose=0,
                n_steps=128, batch_size=128, n_epochs=4, learning_rate=3e-4,
                gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
                policy_kwargs=dict(net_arch=[64, 64]),
            )
            model.learn(total_timesteps=steps)
            env = make_benchmark("tiny", fully_obs=fully_obs)
            for deterministic in (True, False):
                row = dict(
                    obs_mode="fo" if fully_obs else "po",
                    train_steps=steps,
                    eval_mode="argmax" if deterministic else "sampled",
                )
                row.update(_summarise(model, env, deterministic))
                rows.append(row)
                print(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "diag_det_eval.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}")
