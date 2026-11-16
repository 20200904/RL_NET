"""Phase 0 baseline: can SB3 PPO solve a NASim scenario, and how fast?

SB3 is used only as a reference here; the PPO used for the experiments is
implemented by hand in Phase 1.
"""
import csv
import json
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from src.envs.compat import make_benchmark, make_generated

SOLVE_WINDOW = 20        # episodes in the rolling window
SOLVE_THRESHOLD = 0.9    # rolling solve rate that counts as "solved"
EVAL_EPISODES = 20
NUM_ENVS = 4


class EpisodeLogger(gym.Wrapper):
    """Append one row per finished episode to a shared list."""

    def __init__(self, env, log, clock):
        super().__init__(env)
        self._log = log
        self._clock = clock
        self._ret = 0.0
        self._len = 0

    def reset(self, **kwargs):
        self._ret = 0.0
        self._len = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._clock["steps"] += 1
        self._ret += float(reward)
        self._len += 1
        if terminated or truncated:
            self._log.append(dict(
                global_step=self._clock["steps"],
                wall_s=round(time.perf_counter() - self._clock["start"], 2),
                ep_return=self._ret,
                ep_len=self._len,
                solved=int(terminated),
            ))
        return obs, reward, terminated, truncated, info


def _make_env(scenario, fully_obs, net_seed, step_limit):
    if scenario.startswith("gen"):
        return make_generated(
            int(scenario[3:]), seed=net_seed, fully_obs=fully_obs,
            step_limit=step_limit,
        )
    return make_benchmark(scenario, fully_obs=fully_obs)


def _steps_to_solve(episodes):
    window = deque(maxlen=SOLVE_WINDOW)
    for ep in episodes:
        window.append(ep["solved"])
        if len(window) == SOLVE_WINDOW and np.mean(window) >= SOLVE_THRESHOLD:
            return ep["global_step"], ep["wall_s"]
    return None, None


def _evaluate(model, env, deterministic, max_steps):
    solved = 0
    for _ in range(EVAL_EPISODES):
        obs, _ = env.reset()
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                solved += int(terminated)
                break
    return solved / EVAL_EPISODES


def run_one(scenario, fully_obs, seed, total_steps, out_dir, net_seed=1,
            step_limit=400):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    obs_mode = "fo" if fully_obs else "po"
    run_dir = out_dir / f"sb3_{scenario}_{obs_mode}_seed{seed}"
    done_path = run_dir / "done.json"
    if done_path.exists():
        print(f"skip {run_dir.name} (done.json exists)")
        return json.loads(done_path.read_text())
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(1)
    episodes = []
    clock = dict(steps=0, start=time.perf_counter())
    vec_env = DummyVecEnv([
        (lambda: EpisodeLogger(
            _make_env(scenario, fully_obs, net_seed, step_limit),
            episodes, clock))
        for _ in range(NUM_ENVS)
    ])
    model = PPO(
        "MlpPolicy", vec_env, seed=seed, device="cpu", verbose=0,
        n_steps=128, batch_size=128, n_epochs=4, learning_rate=3e-4,
        gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
        policy_kwargs=dict(net_arch=[64, 64]),
    )
    model.learn(total_timesteps=total_steps)
    train_wall = time.perf_counter() - clock["start"]
    train_steps = clock["steps"]

    eval_env = _make_env(scenario, fully_obs, net_seed, step_limit)
    max_steps = eval_env.unwrapped.scenario.step_limit or 1000
    solve_step, solve_wall = _steps_to_solve(episodes)
    last = episodes[-SOLVE_WINDOW:]
    summary = dict(
        scenario=scenario,
        obs_mode=obs_mode,
        seed=seed,
        net_seed=net_seed if scenario.startswith("gen") else "",
        train_steps=train_steps,
        train_wall_s=round(train_wall, 1),
        train_steps_per_sec=round(train_steps / train_wall, 1),
        episodes=len(episodes),
        steps_to_solve="" if solve_step is None else solve_step,
        wall_s_to_solve="" if solve_wall is None else solve_wall,
        final_train_solve_rate=(
            round(float(np.mean([e["solved"] for e in last])), 3) if last else ""
        ),
        final_train_ep_len=(
            round(float(np.mean([e["ep_len"] for e in last])), 1) if last else ""
        ),
        eval_det_solve_rate=_evaluate(model, eval_env, True, max_steps),
        eval_stoch_solve_rate=_evaluate(model, eval_env, False, max_steps),
    )

    with (run_dir / "episodes.csv").open("w", newline="") as f:
        fields = ["global_step", "wall_s", "ep_return", "ep_len", "solved"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(episodes)
    done_path.write_text(json.dumps(summary, indent=2))
    print(summary)
    return summary


def run(scenario, fully_obs, seeds, total_steps, out_dir=Path("results/phase0"),
        net_seed=1, step_limit=400):
    summaries = [
        run_one(scenario, fully_obs, seed, total_steps, out_dir, net_seed,
                step_limit)
        for seed in seeds
    ]
    # rebuild the summary table from every finished run on disk
    rows = [
        json.loads(p.read_text())
        for p in sorted(out_dir.glob("sb3_*/done.json"))
    ]
    out_path = out_dir / "sb3_summary.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}")
    return summaries
