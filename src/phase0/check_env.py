"""check-env: load NASim scenarios, run a random policy, report steps/second."""
import csv
import platform
import time
from pathlib import Path

import gymnasium
import nasim
import numpy as np
import torch

from src.envs.compat import make_benchmark, make_generated


def _random_rollout(env, num_steps, seed):
    env.action_space.seed(seed)
    env.reset(seed=seed)
    episodes = solved = 0
    start = time.perf_counter()
    for _ in range(num_steps):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            episodes += 1
            solved += int(terminated)
            env.reset()
    elapsed = time.perf_counter() - start
    return num_steps / elapsed, episodes, solved


def run(num_steps=1000, out_dir=Path("results/phase0"), seed=0):
    torch.set_num_threads(1)
    print(f"python {platform.python_version()} on {platform.system()}")
    print(f"nasim {nasim.__version__}, gymnasium {gymnasium.__version__}, "
          f"numpy {np.__version__}, torch {torch.__version__}")

    cases = [
        ("tiny", lambda: make_benchmark("tiny")),
        ("small", lambda: make_benchmark("small")),
        ("gen6", lambda: make_generated(6, seed=seed)),
        ("gen8", lambda: make_generated(8, seed=seed)),
    ]
    rows = []
    for name, factory in cases:
        env = factory()
        sps, episodes, solved = _random_rollout(env, num_steps, seed)
        row = dict(
            scenario=name,
            obs_dim=env.observation_space.shape[0],
            num_actions=int(env.action_space.n),
            random_steps=num_steps,
            steps_per_sec=round(sps, 1),
            episodes_finished=episodes,
            episodes_solved=solved,
        )
        rows.append(row)
        print(row)
        env.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "check_env.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}")
