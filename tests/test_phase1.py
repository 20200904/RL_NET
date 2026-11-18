import dataclasses
import json
import os
import subprocess
import sys

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from nasim.envs import NASimEnv
from nasim.envs.action import Exploit, OSScan, PrivilegeEscalation, ServiceScan

from src import runner
from src.agents.ppo import PPOConfig, train
from src.envs import generate as g
from src.envs.cost_wrapper import attack_cost
from src.envs.multi_env import MultiNetEnv

REPO_ROOT = Path(__file__).resolve().parents[1]
OOS = frozenset({(3, 0)})
FRAGILE = frozenset({(3, 1), (3, 0)})
RISKY = g.RISKY_EXPLOITS


def _exploit(name, target):
    service, os, access = g.EXPLOITS[name]
    return Exploit(name, target, cost=1, service=service, os=os, access=access)


# ---- cost rules on hand-made cases ----

@pytest.mark.parametrize("action, expected", [
    # risky exploit on a fragile host -> C2
    (_exploit("e_srv_1", (3, 1)), (0, 1)),
    # safe exploit on the same fragile host -> free
    (_exploit("e_srv_0", (3, 1)), (0, 0)),
    # risky exploit on a robust host -> free
    (_exploit("e_srv_1", (2, 0)), (0, 0)),
    # safe exploit on the out-of-scope host -> C1
    (_exploit("e_srv_0", (3, 0)), (1, 0)),
    # risky exploit on a host that is out of scope and fragile -> both
    (_exploit("e_srv_3_os_1", (3, 0)), (1, 1)),
    # privilege escalation on the out-of-scope host -> C1
    (PrivilegeEscalation("pe_proc_0", (3, 0), cost=1, access=2, process="proc_0"),
     (1, 0)),
    # privilege escalation on a fragile host is not a risky exploit -> free
    (PrivilegeEscalation("pe_proc_0", (3, 1), cost=1, access=2, process="proc_0"),
     (0, 0)),
    # scans are always free, even out of scope
    (ServiceScan((3, 0), cost=1), (0, 0)),
    (OSScan((3, 0), cost=1), (0, 0)),
])
def test_attack_cost_cases(action, expected):
    assert attack_cost(action, OOS, FRAGILE, RISKY) == expected


# ---- generated networks ----

@pytest.fixture(scope="module")
def specs():
    out, _ = g.build_splits(6, sizes={"train": 4, "test_a": 2, "test_b": 2})
    return out


def _find(env, predicate):
    for idx, action in enumerate(env.unwrapped.action_space.actions):
        if predicate(action):
            return idx
    raise LookupError("no such action")


def test_wrapper_charges_attempts_and_keeps_reward(specs):
    spec = specs[0]
    env = g.make_env(spec, step_limit=50)
    raw = NASimEnv(g.make_scenario(spec.seed, spec.num_hosts), fully_obs=False)
    env.reset()
    raw.reset()
    fragile, oos = spec.fragile[0], spec.out_of_scope[0]
    risky = _find(env, lambda a: a.name == "e_srv_1" and a.target == fragile)
    attack_oos = _find(env, lambda a: a.is_exploit() and a.target == oos)
    scan_oos = _find(env, lambda a: a.is_os_scan() and a.target == oos)

    for idx, c1, c2 in [(scan_oos, 0, 0), (attack_oos, 1, 0), (risky, 0, 1)]:
        _, reward, _, _, info = env.step(np.int64(idx))
        _, raw_reward, _, _, _ = raw.step(idx)
        assert (info["cost_c1"], info["cost_c2"]) == (c1, c2)
        assert info["cost"] == c1 + c2
        assert reward == raw_reward      # NASim's reward is untouched


def test_cost_does_not_depend_on_success(specs):
    spec = specs[0]
    env = g.make_env(spec, step_limit=50)
    env.reset()
    # a risky exploit at a fragile host is charged whatever NASim reports
    idx = _find(env, lambda a: a.name == "e_srv_3_os_1"
                and a.target == spec.fragile[-1])
    _, _, _, _, info = env.step(idx)
    assert info["cost_c2"] == 1


def test_step_limit_truncates(specs):
    env = g.make_env(specs[0], step_limit=3)
    env.reset()
    scan = _find(env, lambda a: a.is_os_scan())
    flags = [env.step(scan)[3] for _ in range(3)]
    assert flags == [False, False, True]


def test_observation_remembers_scans_and_hides_fragility(specs):
    spec = specs[0]
    env = g.make_env(spec, step_limit=50)
    obs, _ = env.reset()
    dmz = (1, 0)
    assert not env.os_known(dmz)
    _, _, _, _, info = env.step(_find(env, lambda a: a.is_os_scan() and a.target == dmz))
    assert info["target_os_known"] is False          # knowledge before the scan
    assert env.os_known(dmz)
    # a later, unrelated action must not erase the scan result
    obs, _, _, _, info = env.step(
        _find(env, lambda a: a.is_service_scan() and a.target == dmz))
    assert env.os_known(dmz) and env.services_known(dmz)
    # the last num_hosts entries are the out-of-scope flags, nothing else
    flags = obs[-spec.num_hosts:]
    assert flags.sum() == 1
    assert flags[env.host_row[spec.out_of_scope[0]]] == 1
    assert obs.shape == env.observation_space.shape


def test_all_networks_share_spaces(specs):
    shapes = {
        (g.make_env(s, 10).observation_space.shape, int(g.make_env(s, 10).action_space.n))
        for s in specs
    }
    assert len(shapes) == 1


def test_split_seeds_disjoint_and_safe_path_exists(specs):
    by_split = {}
    for spec in specs:
        by_split.setdefault(spec.split, set()).add(spec.seed)
        assert spec.safe_path_len is not None and spec.safe_path_len > 0
        assert spec.out_of_scope[0] not in g.make_scenario(
            spec.seed, spec.num_hosts).sensitive_addresses
    assert not by_split["train"] & by_split["test_a"]
    assert not by_split["train"] & by_split["test_b"]
    assert not by_split["test_a"] & by_split["test_b"]


@pytest.mark.parametrize("num_hosts", [6, 8])
def test_saved_splits_are_disjoint(num_hosts):
    if not g.splits_path(num_hosts).exists():
        pytest.skip("splits not generated")
    splits = g.load_splits(num_hosts)
    seeds = {name: {s.seed for s in part} for name, part in splits.items()}
    assert len(seeds["train"]) == 32
    assert len(seeds["test_a"]) == len(seeds["test_b"]) == 30
    assert not seeds["train"] & (seeds["test_a"] | seeds["test_b"])
    assert not seeds["test_a"] & seeds["test_b"]


def test_networks_identical_across_processes():
    # Regression test: NASim's firewall generation depends on the per-process
    # string hash seed. The same network seed must give the same network in
    # every process, or train and test sets silently change between runs.
    code = ("from src.envs import generate as g;"
            "print([g.digest(g.make_scenario(s, 8), ()) for s in range(10000, 10020)])")
    outputs = set()
    for hash_seed in ("1", "2", "3"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        outputs.add(subprocess.run(
            [sys.executable, "-W", "ignore", "-c", code], env=env,
            capture_output=True, text=True, check=True).stdout)
    assert len(outputs) == 1


def test_train_and_eval_identical_across_processes(tmp_path):
    # The in-process test below (test_same_seed_same_result) cannot see
    # anything that depends on the process, such as the string hash seed.
    # This one runs a short train + eval job in two separate processes with
    # different hash seeds and compares what they write.
    #
    # A job this short barely touches the firewall, so equal outputs alone
    # are weak evidence: with NASim's original firewall code, hash seeds 1
    # and 2 build different networks yet produced identical eval.csv and
    # training logs. The child therefore also records the digest of every
    # network it actually builds, and those are compared too.
    if not g.splits_path(8).exists():
        pytest.skip("splits not generated")
    hash_seeds = ("1", "2")
    # precondition: the bug can only show if the two processes iterate a set
    # of service names in different orders
    orders = {
        subprocess.run(
            [sys.executable, "-c", "print(list({'srv_0','srv_1','srv_2','srv_3'}))"],
            env=dict(os.environ, PYTHONHASHSEED=h), capture_output=True,
            text=True, check=True).stdout
        for h in hash_seeds
    }
    assert len(orders) == 2, "pick hash seeds that order string sets differently"

    code = (
        "import json, sys\n"
        "from src.envs import generate as g\n"
        "built, make = [], g.make_scenario\n"
        "def recording_make(seed, num_hosts):\n"
        "    scenario = make(seed, num_hosts)\n"
        "    built.append([seed, g.digest(scenario, ())])\n"
        "    return scenario\n"
        "g.make_scenario = recording_make\n"
        "from src import runner\n"
        "runner.run_job(dict(method='M1', num_hosts=8, k=2, seed=5,"
        " out=sys.argv[1], step_limit=30, total_steps=1024,"
        " ppo=dict(num_envs=4, num_steps=64), method_params=dict(algo='ppo'),"
        " eval_episodes=1, scenario_root='scenarios'))\n"
        "open(sys.argv[1] + '/networks.json', 'w').write(json.dumps(built))\n"
    )
    outputs = []
    for hash_seed in hash_seeds:
        out = tmp_path / hash_seed
        proc = subprocess.run(
            [sys.executable, "-W", "ignore", "-c", code, str(out)],
            env=dict(os.environ, PYTHONHASHSEED=hash_seed), cwd=REPO_ROOT,
            capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr[-2000:]
        run_dir = out / "M1_h8_K2_seed5"
        outputs.append(dict(
            networks=json.loads((out / "networks.json").read_text()),
            eval=(run_dir / "eval.csv").read_text(),
            train=pd.read_csv(run_dir / "episodes.csv").drop(columns="wall_s"),
        ))
    first, second = outputs
    # train (2) + test A (30) + test B (30) networks
    assert len({seed for seed, _ in first["networks"]}) == 62
    assert first["networks"] == second["networks"]
    assert first["eval"] == second["eval"]
    assert first["train"].equals(second["train"])
    assert len(first["train"]) > 0


def test_digest_covers_everything_that_defines_a_network(specs):
    spec = specs[0]
    scenario = g.make_scenario(spec.seed, spec.num_hosts)
    described = g.describe(scenario, spec.out_of_scope)
    assert set(described) >= {
        "hosts", "firewall", "exploits", "privescs", "sensitive_hosts",
        "out_of_scope", "fragile", "fragile_rules", "risky_exploits",
        "subnets", "topology",
    }
    assert {"os", "services", "processes"} <= set(described["hosts"][0])
    base = g.digest(scenario, spec.out_of_scope)
    # a different out-of-scope host is a different network
    other = [a for a in scenario.address_space if a not in spec.out_of_scope][0]
    assert g.digest(scenario, (other,)) != base
    # so is a firewall that lets one more service through
    pair = sorted(scenario.firewall)[0]
    scenario.firewall[pair] = set(scenario.firewall[pair]) ^ {"srv_3"}
    assert g.digest(scenario, spec.out_of_scope) != base


def test_digest_mismatch_is_detected(specs):
    wrong = dataclasses.replace(specs[0], digest="0" * 16)
    with pytest.raises(RuntimeError):
        g.verified_scenario(wrong)
    assert g.verified_scenario(specs[0]) is not None


def test_fragile_fraction_shift(specs):
    frac = lambda s: len(s.fragile) / s.num_hosts
    assert max(frac(s) for s in specs if s.split != "test_b") \
        < min(frac(s) for s in specs if s.split == "test_b")


# ---- training ----

def _short_train(specs, seed):
    cfg = PPOConfig(total_steps=2 * 4 * 32, num_envs=4, num_steps=32, seed=seed)
    fns = [(lambda i=i: MultiNetEnv(specs[:4], 30, seed=seed * 1000 + i))
           for i in range(cfg.num_envs)]
    agent, summary = train(fns, cfg)
    return torch.cat([p.flatten() for p in agent.parameters()]), summary


def test_same_seed_same_result(specs):
    params_a, summary_a = _short_train(specs, seed=3)
    params_b, summary_b = _short_train(specs, seed=3)
    params_c, _ = _short_train(specs, seed=4)
    assert torch.equal(params_a, params_b)
    assert summary_a["train_episodes"] == summary_b["train_episodes"]
    assert not torch.equal(params_a, params_c)


def test_runner_skips_finished_run(tmp_path):
    job = dict(method="M1", num_hosts=6, k=4, seed=0, out=str(tmp_path))
    run_dir = tmp_path / runner.run_name(job)
    run_dir.mkdir()
    (run_dir / "done.json").write_text(json.dumps({}))
    # no other job fields are needed: a finished run must not be touched
    assert runner.run_job(job) == dict(run=run_dir.name, skipped=True)
