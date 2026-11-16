"""Resumable experiment runner: one job = train one policy, then evaluate it.

A job writes into ``<out>/<run name>/`` and finishes by writing ``done.json``.
Jobs whose ``done.json`` exists are skipped, so an interrupted grid can simply
be started again. A job that died half-way is rerun from the beginning (runs
are short), which keeps every finished run exactly reproducible from its seed.
"""
import json
import multiprocessing as mp
import time
from pathlib import Path

import yaml

from src.agents.ppo import PPOConfig, train
from src.agents.random_agent import RandomAgent
from src.envs.generate import load_splits
from src.envs.multi_env import MultiNetEnv
from src.eval.run_eval import evaluate, write_rows


def load_config(path):
    # explicit encoding: the Windows default (cp949 here) is not UTF-8
    return yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))


def run_name(job):
    return f'{job["method"]}_h{job["num_hosts"]}_K{job["k"]}_seed{job["seed"]}'


def expand_jobs(cfg):
    """Turn a config into the flat list of jobs of its grid."""
    jobs = []
    for num_hosts in cfg["grid"]["num_hosts"]:
        for method in cfg["grid"]["methods"]:
            for k in cfg["grid"]["k"]:
                for seed in cfg["grid"]["seeds"]:
                    jobs.append(dict(
                        method=method, num_hosts=num_hosts, k=k, seed=seed,
                        out=cfg["out"],
                        step_limit=cfg["step_limit"][num_hosts],
                        total_steps=cfg["total_steps"][num_hosts],
                        ppo=cfg["ppo"],
                        method_params=cfg["methods"][method],
                        eval_episodes=cfg["eval"]["sampled_episodes_per_network"],
                        scenario_root=cfg.get("scenario_root", "scenarios"),
                    ))
    return jobs


def run_job(job):
    run_dir = Path(job["out"]) / run_name(job)
    done_path = run_dir / "done.json"
    if done_path.exists():
        return dict(run=run_dir.name, skipped=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    splits = load_splits(job["num_hosts"], Path(job["scenario_root"]))
    train_specs = splits["train"][:job["k"]]
    step_limit = job["step_limit"]
    params = job["method_params"]
    seed = job["seed"]

    summary = {}
    if params["algo"] == "random":
        probe = MultiNetEnv(train_specs[:1], step_limit)
        agent = RandomAgent(int(probe.action_space.n), seed=seed)
    elif params["algo"] == "ppo":
        cfg = PPOConfig(
            total_steps=job["total_steps"], seed=seed,
            cost_penalty=params.get("cost_penalty", 0.0), **job["ppo"],
        )
        env_fns = [
            (lambda i=i: MultiNetEnv(train_specs, step_limit, seed=seed * 1000 + i))
            for i in range(cfg.num_envs)
        ]
        agent, summary = train(env_fns, cfg, run_dir)
    else:
        raise ValueError(f'unknown algo {params["algo"]}')

    rows = []
    eval_sets = dict(train=train_specs, test_a=splits["test_a"],
                     test_b=splits["test_b"])
    for split, specs in eval_sets.items():
        rows += evaluate(agent, specs, split, step_limit, job["eval_episodes"],
                         deterministic=False, seed=seed)
        if params["algo"] != "random":
            rows += evaluate(agent, specs, split, step_limit, 1,
                             deterministic=True, seed=seed)
    write_rows(rows, run_dir / "eval.csv")

    meta = {k: v for k, v in job.items() if k != "out"}
    meta.update(train=summary, wall_s=round(time.perf_counter() - start, 1))
    done_path.write_text(json.dumps(meta, indent=2))
    return dict(run=run_dir.name, skipped=False, wall_s=meta["wall_s"])


def run_grid(jobs, workers=1):
    """Run jobs, at most ``workers`` (<= 3 on the 8 GB laptop) at a time."""
    workers = max(1, min(workers, 3))
    start = time.perf_counter()
    if workers == 1:
        results = []
        for job in jobs:
            results.append(run_job(job))
            print(results[-1], flush=True)
    else:
        results = []
        with mp.get_context("spawn").Pool(workers, maxtasksperchild=1) as pool:
            for result in pool.imap_unordered(run_job, jobs):
                results.append(result)
                print(result, flush=True)
    print(f"{len(results)} jobs, {time.perf_counter() - start:.0f}s total")
    return results
