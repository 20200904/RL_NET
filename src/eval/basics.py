"""Basic checks before changing the design (no training).

1. rule agents: is the task solvable from observations alone, with and
   without cost?
2. test A solve rate of M1 at 0, 1M and 2M training steps
3. what one scan / one failed exploit shows in the observation
4. action costs and rewards; blind exploiting vs scanning first

Writes into results/basics/.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.agents.ppo import Agent
from src.agents.rule_agent import MODES, RuleAgent
from src.envs.generate import EXPLOITS, load_splits, make_env, make_scenario
from src.envs.multi_env import MultiNetEnv
from src.eval.run_eval import evaluate
from src.eval.stats import summarise_episodes

OUT = Path("results/basics")
NUM_HOSTS, STEP_LIMIT = 8, 100
CONVERGE = Path("results/phase1/converge")
HOST_FEATURES = ["compromised", "reachable", "discovered", "value",
                 "discovery_value", "access"]


# ---- 1. rule agents ----

def rule_agents(splits):
    probe = make_env(splits["train"][0], STEP_LIMIT)
    rows = []
    for mode in MODES:
        agent = RuleAgent(probe, mode)
        for split, specs in splits.items():
            # agent and environment are both deterministic: one episode each
            eps = pd.DataFrame(evaluate(agent, specs, split, STEP_LIMIT, 1,
                                        deterministic=True))
            row = dict(mode=mode, split=split)
            row.update(summarise_episodes(eps))
            row.update(mean_steps_all=eps["steps"].mean(),
                       mean_return=eps["ep_return"].mean())
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "rule_agents.csv", index=False)
    return df


# ---- 2. test A solve rate over training steps ----

def _initial_policy(specs, seed, hidden):
    """The exact untrained network a run starts from (same seeding as train)."""
    env = MultiNetEnv(specs[:1], STEP_LIMIT)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return Agent(env.observation_space.shape[0], int(env.action_space.n), hidden)


def test_a_over_steps(splits):
    rows = []
    for k in (16, 32):
        for seed in (0, 1, 2):
            meta = json.loads(
                (CONVERGE / "2m" / f"M1_h8_K{k}_seed{seed}" / "done.json").read_text())
            agent = _initial_policy(splits["train"][:k], seed, meta["ppo"]["hidden"])
            for split, specs in (("train", splits["train"][:k]),
                                 ("test_a", splits["test_a"])):
                eps = pd.DataFrame(evaluate(agent, specs, split, STEP_LIMIT,
                                            meta["eval_episodes"],
                                            deterministic=False, seed=seed))
                rows.append(dict(k=k, seed=seed, step=0, split=split,
                                 solve_rate=eps["solved"].mean()))
            for label, step in (("1m", 1_000_000), ("2m", 2_000_000)):
                eps = pd.read_csv(CONVERGE / label / f"M1_h8_K{k}_seed{seed}" / "eval.csv")
                eps = eps[eps["mode"] == "sampled"]
                for split in ("train", "test_a"):
                    rows.append(dict(k=k, seed=seed, step=step, split=split,
                                     solve_rate=eps.loc[eps["split"] == split,
                                                        "solved"].mean()))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "test_a_over_steps.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, k in zip(axes, (16, 32)):
        curves = pd.read_csv(CONVERGE / "2m" / "curve_bins.csv")
        for i, (seed, c) in enumerate(curves[curves["k"] == k].groupby("seed")):
            ax.plot(c["step_bin"] / 1e6, c["solve_rate"], color=f"C{i}", alpha=0.35,
                    label="training pool, during training" if i == 0 else None)
            pts = df[(df["k"] == k) & (df["seed"] == seed) & (df["split"] == "test_a")]
            ax.plot(pts["step"] / 1e6, pts["solve_rate"], "o-", color=f"C{i}",
                    label=f"test A, seed {seed}")
        ax.set_title(f"M1, K={k}: only checkpoints at 0, 1M, 2M exist")
        ax.set_xlabel("environment steps (millions)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("solve rate (sampled)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "test_a_over_steps.png", dpi=150)
    return df


# ---- 3. observation content and failure modes ----

def _row(env, raw_obs, addr):
    rows = raw_obs.reshape(env.num_hosts + 1, env.host_vec_size)
    return rows[env.host_row[addr]], rows[-1][:4]


def _describe_row(env, vec):
    scenario = env.unwrapped.scenario
    base = env.os_slice.start - len(HOST_FEATURES)
    out = []
    if vec[:base].any():
        out.append("address")
    out += [f"{n}={vec[base + i]:g}" for i, n in enumerate(HOST_FEATURES)
            if vec[base + i]]
    out += [f"os:{n}" for n, x in zip(scenario.os, vec[env.os_slice]) if x]
    out += [f"service:{n}" for n, x in zip(scenario.services, vec[env.srv_slice]) if x]
    proc = vec[env.srv_slice.stop:]
    out += [f"process:{n}" for n, x in zip(scenario.processes, proc) if x]
    return ", ".join(out) or "(nothing)"


def _aux(aux):
    names = ["success", "connection_error", "permission_error", "undefined_error"]
    return ", ".join(n for n, x in zip(names, aux) if x) or "(no flag)"


def observation_probe(splits):
    """Step a real network and record what the raw NASim observation shows.

    The raw observation (before ObsMemory) is what NASim reveals for the last
    action; ObsMemory only keeps the maximum of these over the episode.
    """
    rows = []

    def act(env, label, kind, target, name=None):
        idx = next(i for i, a in enumerate(env.unwrapped.action_space.actions)
                   if tuple(a.target) == target and (
                       (kind == "exploit" and a.is_exploit() and a.name == name)
                       or (kind == "privesc" and a.is_privilege_escalation() and a.name == name)
                       or a.name == kind))
        env.step(idx)
        raw = env.unwrapped.last_obs.numpy_flat()
        vec, aux = _row(env, raw, target)
        rows.append(dict(case=label, action=f"{kind} {name or ''} -> {target}",
                         result_flags=_aux(aux),
                         target_row_shows=_describe_row(env, vec)))

    # a network where an exploit that fits the host is blocked by the firewall
    for spec in splits["train"]:
        scenario = make_scenario(spec.seed, spec.num_hosts)
        dmz = scenario.hosts[(1, 0)]
        first = next(n for n, (srv, os, _) in EXPLOITS.items()
                     if dmz.services[srv] and (os is None or dmz.os[os]))
        blocked = None
        for addr, host in scenario.hosts.items():
            if addr[0] == 1:
                continue
            for name, (srv, os, _) in EXPLOITS.items():
                fits = host.services[srv] and (os is None or host.os[os])
                allowed = {s for (src, dst), sv in scenario.firewall.items()
                           if dst == addr[0] and src in (0, 1) for s in sv}
                if fits and srv not in allowed and scenario.topology[1][addr[0]]:
                    blocked = (addr, name)
            if blocked:
                break
        if blocked:
            break
    env = make_env(spec, STEP_LIMIT)
    env.reset()
    act(env, "OS scan of the DMZ host", "os_scan", (1, 0))
    act(env, "service scan of the DMZ host", "service_scan", (1, 0))
    wrong = next(n for n, (srv, os, _) in EXPLOITS.items()
                 if not dmz.services[srv])
    act(env, "exploit whose service the host does not run", "exploit", (1, 0), wrong)
    target = blocked[0]
    act(env, "exploit on a host not yet discovered", "exploit", target, blocked[1])
    act(env, "exploit that fits the DMZ host", "exploit", (1, 0), first)
    act(env, "process scan of the compromised DMZ host", "process_scan", (1, 0))
    act(env, "subnet scan from the DMZ host", "subnet_scan", (1, 0))
    act(env, "exploit that fits the host but the firewall blocks", "exploit",
        target, blocked[1])
    for name, (srv, os, _) in EXPLOITS.items():
        host = scenario.hosts[target]
        if not (host.services[srv] and (os is None or host.os[os])):
            act(env, "exploit that does not fit the same host", "exploit", target, name)
            break
    df = pd.DataFrame(rows)
    df.insert(0, "network_seed", spec.seed)
    df.to_csv(OUT / "observation_probe.csv", index=False)
    return df


# ---- 4. costs and rewards ----

def cost_table(splits):
    scenario = make_scenario(splits["train"][0].seed, NUM_HOSTS)
    rows = [dict(action=a, cost=getattr(scenario, f"{a}_cost"), reward="")
            for a in ("service_scan", "os_scan", "subnet_scan", "process_scan")]
    rows += [dict(action=f"exploit {n}", cost=e["cost"],
                  reward=f"gives access {e['access']}")
             for n, e in scenario.exploits.items()]
    rows += [dict(action=f"privesc {n}", cost=p["cost"], reward="gives root")
             for n, p in scenario.privescs.items()]
    values = sorted({float(h.value) for h in scenario.hosts.values()})
    rows.append(dict(action="root on a sensitive host (once)", cost="",
                     reward=float(max(scenario.sensitive_hosts.values()))))
    rows.append(dict(action="root on any other host (once)", cost="",
                     reward=min(values)))
    rows.append(dict(action="subnet scan, per newly discovered host", cost="",
                     reward=float(next(iter(scenario.hosts.values())).discovery_value)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "costs_rewards.csv", index=False)

    # attempts to land the first exploit on each host, from the true config,
    # ignoring the firewall: blind (table order) vs scan first (2 + 1)
    tries = []
    for spec in splits["train"]:
        sc = make_scenario(spec.seed, spec.num_hosts)
        for addr, host in sc.hosts.items():
            fits = [n for n, (srv, os, _) in EXPLOITS.items()
                    if host.services[srv] and (os is None or host.os[os])]
            if not fits:
                continue
            blind = list(EXPLOITS).index(fits[0]) + 1
            tries.append(dict(network_seed=spec.seed, host=str(addr),
                              matching_exploits=len(fits), blind_attempts=blind,
                              scan_first_actions=3))
    tries = pd.DataFrame(tries)
    tries.to_csv(OUT / "blind_vs_scan_attempts.csv", index=False)
    return df, tries


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    splits = load_splits(NUM_HOSTS)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)

    print("== 1. rule agents ==")
    r = rule_agents(splits)
    print(r[["mode", "split", "episodes", "solve_rate", "violation_rate",
             "safe_solve_rate", "mean_steps_all", "mean_return",
             "cost_c1_per_episode", "cost_c2_per_episode"]].round(3).to_string(index=False))

    print("\n== 2. M1 solve rate at 0 / 1M / 2M steps ==")
    t = test_a_over_steps(splits)
    print(t.pivot_table(index=["k", "split", "step"], columns="seed",
                        values="solve_rate").round(3).to_string())

    print("\n== 3. observation probe ==")
    print(observation_probe(splits).to_string(index=False))

    print("\n== 4. costs and rewards ==")
    costs, tries = cost_table(splits)
    print(costs.to_string(index=False))
    print(tries[["matching_exploits", "blind_attempts", "scan_first_actions"]]
          .describe().loc[["mean", "min", "max"]].round(2).to_string())
    print(f"wrote csv files and test_a_over_steps.png in {OUT}")
