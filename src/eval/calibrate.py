"""Phase 1 calibration: how often does a random policy solve a network within
a given step limit? Used to pick the step limit (target: <= 20%)."""
import csv
from pathlib import Path

import numpy as np

from src.agents.random_agent import RandomAgent
from src.envs.generate import load_splits
from src.envs.multi_env import MultiNetEnv

LIMITS = (20, 30, 40, 50, 60, 80, 100, 150, 200, 300)
MAX_STEPS = max(LIMITS)


def path_lengths(num_hosts, out_dir=Path("results/phase1"),
                 scenario_root=Path("scenarios")):
    """Per network: shortest path with and without the zero-cost restriction.

    ``shortcut`` is 1 when ignoring the costs gives a strictly shorter path,
    i.e. when the reward-optimal behaviour has to violate a constraint.
    """
    from src.envs.generate import make_scenario, shortest_safe_path

    rows = []
    for split, specs in load_splits(num_hosts, scenario_root).items():
        for spec in specs:
            scenario = make_scenario(spec.seed, spec.num_hosts)
            any_len = shortest_safe_path(scenario, (), ())
            rows.append(dict(
                num_hosts=num_hosts, split=split, network_seed=spec.seed,
                fragile_hosts=len(spec.fragile),
                safe_path_len=spec.safe_path_len,
                unconstrained_path_len=any_len,
                shortcut=int(any_len < spec.safe_path_len),
            ))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"path_lengths_h{num_hosts}.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for split in ("train", "test_a", "test_b"):
        part = [r for r in rows if r["split"] == split]
        print(dict(
            num_hosts=num_hosts, split=split, networks=len(part),
            shortcut_frac=round(float(np.mean([r["shortcut"] for r in part])), 3),
            mean_safe_len=round(float(np.mean([r["safe_path_len"] for r in part])), 2),
            mean_unconstrained_len=round(float(np.mean(
                [r["unconstrained_path_len"] for r in part])), 2),
        ))
    print(f"wrote {out_path}")


def run(num_hosts, episodes_per_network=30, out_dir=Path("results/phase1"),
        scenario_root=Path("scenarios"), seed=0):
    specs = load_splits(num_hosts, scenario_root)["train"]
    env = MultiNetEnv(specs, MAX_STEPS)
    agent = RandomAgent(int(env.action_space.n), seed=seed)
    np.random.seed(seed)

    solve_steps = []    # steps to the goal, or inf if not reached
    for net_idx in range(len(specs)):
        for _ in range(episodes_per_network):
            obs, _ = env.reset(options={"network": net_idx})
            steps = 0
            while True:
                obs, _, terminated, truncated, _ = env.step(agent.act(obs))
                steps += 1
                if terminated or truncated:
                    solve_steps.append(steps if terminated else np.inf)
                    break
    solve_steps = np.array(solve_steps)

    safe_lens = [s.safe_path_len for s in specs]
    rows = [
        dict(
            num_hosts=num_hosts,
            networks=len(specs),
            episodes=len(solve_steps),
            step_limit=limit,
            random_solve_rate=round(float(np.mean(solve_steps <= limit)), 4),
            safe_path_len_min=min(safe_lens),
            safe_path_len_max=max(safe_lens),
        )
        for limit in LIMITS
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"calibration_h{num_hosts}.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"wrote {out_path}")
