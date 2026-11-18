"""Roll a policy out on a fixed set of networks and record one row per episode."""
import csv
from pathlib import Path

import numpy as np
import torch

from src.envs.generate import RISKY_EXPLOITS
from src.envs.multi_env import MultiNetEnv

EVAL_FIELDS = [
    "split", "network_seed", "mode", "episode", "solved", "steps", "ep_return",
    "cost", "cost_c1", "cost_c2",
    "exploits", "exploits_scanned",      # scanned = OS and services known before
    "privescs",
    "risky", "risky_scanned",
    "c2_unscanned",
]


def evaluate(agent, specs, split, step_limit, episodes_per_network,
             deterministic, seed=0):
    """``agent`` needs ``act(obs, deterministic) -> int``."""
    # one thread: multi-threaded float sums can differ in the last bit, which
    # is enough to flip a sampled action and change the whole episode
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = MultiNetEnv(specs, step_limit)
    mode = "argmax" if deterministic else "sampled"
    rows = []
    for net_idx, spec in enumerate(specs):
        for episode in range(episodes_per_network):
            obs, _ = env.reset(options={"network": net_idx})
            if hasattr(agent, "reset"):      # agents with per-episode memory
                agent.reset()
            row = dict.fromkeys(EVAL_FIELDS, 0)
            row.update(split=split, network_seed=spec.seed, mode=mode,
                       episode=episode, ep_return=0.0)
            while True:
                obs, reward, terminated, truncated, info = env.step(
                    agent.act(obs, deterministic)
                )
                row["steps"] += 1
                row["ep_return"] += float(reward)
                row["cost"] += info["cost"]
                row["cost_c1"] += info["cost_c1"]
                row["cost_c2"] += info["cost_c2"]
                row["privescs"] += int(info["action_kind"] == "privesc")
                if info["action_kind"] == "exploit":
                    scanned = (info["target_os_known"]
                               and info["target_services_known"])
                    row["exploits"] += 1
                    row["exploits_scanned"] += int(scanned)
                    if info["action_name"] in RISKY_EXPLOITS:
                        row["risky"] += 1
                        row["risky_scanned"] += int(scanned)
                    if info["cost_c2"] and not scanned:
                        row["c2_unscanned"] += 1
                if terminated or truncated:
                    row["solved"] = int(terminated)
                    break
            rows.append(row)
    return rows


def write_rows(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
