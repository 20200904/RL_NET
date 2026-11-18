"""Reward redesign, stage 1: audit the current reward (no training).

- Δ per network: best return of any path to the goal minus best return of a
  zero-cost path, with full information (exhaustive search over NASim states).
- Whether the return-optimal path needs a C1 event, with and without the
  "+1 for rooting a non-target host" reward.
- C1 events of the converged M1 policies (2M steps), classified by what the
  attempt earned.

Rewards are recomputed from NASim's action results through ``RewardSpec``,
so the same search can score other reward candidates in stage 2. Under the
uniform spec the recomputed reward is checked against NASim's own reward on
every transition.
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.agents.ppo import Agent
from src.envs.cost_wrapper import attack_cost
from src.envs.generate import RISKY_EXPLOITS, load_splits, make_scenario
from src.envs.multi_env import MultiNetEnv
from nasim.envs import NASimEnv

OUT = Path("results/reward")
NUM_HOSTS, STEP_LIMIT = 8, 100
CONVERGE = Path("results/phase1/converge/2m")


@dataclass(frozen=True)
class RewardSpec:
    scan_cost: float = 1.0           # all four scans, subnet scan included
    attack_cost: float = 1.0         # exploits and privilege escalations
    discovery_reward: float = 1.0    # per host newly found by a subnet scan
    nontarget_root_reward: float = 1.0
    target_root_reward: float = 10.0


UNIFORM = RewardSpec()


def reward(spec, action, result, sensitive):
    """Reward of one NASim transition under ``spec``.

    ``result`` is NASim's action-result dict (``ActionResult.info()``). NASim
    pays ``value - action.cost``, where ``value`` is the host value the first
    time root is reached on a host (exploit or privesc) or the sum of
    discovery values for a subnet scan.
    """
    attack = action.is_exploit() or action.is_privilege_escalation()
    r = -(spec.attack_cost if attack else spec.scan_cost)
    if action.is_subnet_scan() and result["success"]:
        r += spec.discovery_reward * sum(result["newly_discovered"].values())
    elif attack and result["success"] and result["value"] > 0:
        r += (spec.target_root_reward if tuple(action.target) in sensitive
              else spec.nontarget_root_reward)
    return r


class PathSearch:
    """Best undiscounted return from the initial state to the goal.

    Only exploits, privilege escalations and subnet scans can change NASim's
    state; OS, service and process scans never do, so with full information
    they only cost and never lie on an optimal path. States only gain access
    and discovered hosts, so the state graph has no cycles and the best
    return is a longest path, found by memoised depth-first search.
    """

    def __init__(self, spec_network, specs=(UNIFORM,)):
        self.scenario = make_scenario(spec_network.seed, spec_network.num_hosts)
        self.env = NASimEnv(self.scenario, fully_obs=True)
        self.sensitive = set(self.scenario.sensitive_addresses)
        self.oos = frozenset(spec_network.out_of_scope)
        self.fragile = frozenset(spec_network.fragile)
        self.actions = [
            a for a in self.env.action_space.actions
            if a.is_exploit() or a.is_privilege_escalation() or a.is_subnet_scan()
        ]
        self.cost = {id(a): attack_cost(a, self.oos, self.fragile, RISKY_EXPLOITS)
                     for a in self.actions}
        self.specs = specs
        self.env.reset()
        self.start = self.env.current_state
        self._edges = {}

    def _edges_from(self, state):
        """State-changing transitions: (action, result, next_state, key, done)."""
        key = state.tensor.tobytes()
        if key not in self._edges:
            edges = []
            for a in self.actions:
                nxt, _, nasim_r, done, result = self.env.generative_step(state, a)
                nkey = nxt.tensor.tobytes()
                if nkey == key:
                    continue
                edges.append((a, nxt, nkey, done, nasim_r, result))
            self._edges[key] = edges
        return self._edges[key]

    def best(self, spec, allow=lambda c1, c2: True):
        """Best return and path length under ``spec`` using only actions whose
        (c1, c2) cost passes ``allow``. Returns (return, length) or (None, None)
        when the goal cannot be reached that way."""
        memo = {}

        def go(state, key):
            if key in memo:
                return memo[key]
            best = (None, None)
            for a, nxt, nkey, done, nasim_r, result in self._edges_from(state):
                if not allow(*self.cost[id(a)]):
                    continue
                r = reward(spec, a, result, self.sensitive)
                if spec == UNIFORM and not np.isclose(r, nasim_r):
                    raise AssertionError(f"reward {r} != NASim {nasim_r} for {a}")
                if done:
                    cand = (r, 1)
                else:
                    sub = go(nxt, nkey)
                    if sub[0] is None:
                        continue
                    cand = (r + sub[0], 1 + sub[1])
                # higher return first, then the shorter path
                if best[0] is None or (cand[0], -cand[1]) > (best[0], -best[1]):
                    best = cand
            memo[key] = best
            return best

        return go(self.start, self.start.tensor.tobytes())


def _no_cost(c1, c2):
    return c1 == 0 and c2 == 0


def _no_c1(c1, c2):
    return c1 == 0


def _audit_network(item):
    split, net = item
    torch.set_num_threads(1)
    no_nontarget = RewardSpec(nontarget_root_reward=0.0)
    search = PathSearch(net)
    best, best_len = search.best(UNIFORM)
    safe, safe_len = search.best(UNIFORM, _no_cost)
    no_c1_ret, _ = search.best(UNIFORM, _no_c1)
    best0, _ = search.best(no_nontarget)
    no_c1_0, _ = search.best(no_nontarget, _no_c1)
    return dict(
        split=split, network_seed=net.seed,
        best_return=best, best_len=best_len,
        safe_return=safe, safe_len=safe_len,
        delta=best - safe,
        optimum_needs_c1=int(no_c1_ret < best),
        optimum_needs_c1_without_nontarget_reward=int(no_c1_0 < best0),
    )


def optimal_paths(splits, workers=3):
    import multiprocessing as mp
    items = [(split, net) for split, specs in splits.items() for net in specs]
    with mp.get_context("spawn").Pool(workers) as pool:
        rows = pool.map(_audit_network, items, chunksize=1)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "audit_optimal_paths.csv", index=False)
    return df


# ---- C1 events of the converged M1 policies ----

def trace_m1(splits):
    """Replay M1's evaluation (same seeding as run_eval.evaluate) step by step
    and record every C1 event with what it earned."""
    torch.set_num_threads(1)
    rows, check = [], []
    for k in (16, 32):
        for seed in (0, 1, 2):
            run_dir = CONVERGE / f"M1_h8_K{k}_seed{seed}"
            meta = json.loads((run_dir / "done.json").read_text())
            sets = dict(train=splits["train"][:k], test_a=splits["test_a"],
                        test_b=splits["test_b"])
            probe = MultiNetEnv(sets["train"][:1], STEP_LIMIT)
            agent = Agent(probe.observation_space.shape[0],
                          int(probe.action_space.n), meta["ppo"]["hidden"])
            agent.load_state_dict(torch.load(run_dir / "model.pt"))
            for split, specs in sets.items():
                torch.manual_seed(seed)
                np.random.seed(seed)
                env = MultiNetEnv(specs, STEP_LIMIT)
                c1_total = 0
                for net_idx, spec in enumerate(specs):
                    oos = spec.out_of_scope[0]
                    for episode in range(meta["eval_episodes"]):
                        obs, _ = env.reset(options={"network": net_idx})
                        used_as_pivot, events = False, []
                        while True:
                            obs, r, term, trunc, info = env.step(agent.act(obs, False))
                            if info["cost_c1"]:
                                events.append(dict(
                                    kind=info["action_kind"],
                                    success=bool(info["success"]),
                                    earned_root_value=bool(info["success"]
                                                           and info["value"] > 0),
                                ))
                            if (info["action_kind"] == "subnet_scan"
                                    and info["target"] == oos and info["success"]):
                                used_as_pivot = True
                            if term or trunc:
                                break
                        for e in events:
                            e.update(k=k, seed=seed, split=split,
                                     network_seed=spec.seed, episode=episode,
                                     episode_solved=int(term),
                                     oos_used_for_subnet_scan=int(used_as_pivot))
                        rows += events
                        c1_total += len(events)
                # must equal the C1 total in the run's eval.csv
                ev = pd.read_csv(run_dir / "eval.csv")
                expected = ev[(ev["mode"] == "sampled") & (ev["split"] == split)]["cost_c1"].sum()
                check.append(dict(k=k, seed=seed, split=split,
                                  traced_c1=c1_total, eval_csv_c1=int(expected)))
    events = pd.DataFrame(rows)
    events.to_csv(OUT / "audit_m1_c1_events.csv", index=False)
    pd.DataFrame(check).to_csv(OUT / "audit_m1_trace_check.csv", index=False)
    return events, pd.DataFrame(check)


def classify(events):
    def outcome(e):
        if e["earned_root_value"]:
            return "earned the +1 (root on the out-of-scope host)"
        if e["success"]:
            return "succeeded without the +1 (user access)"
        return "failed attempt"
    events = events.assign(outcome=events.apply(outcome, axis=1))
    table = (events.groupby(["k", "split", "outcome"]).size()
             .unstack(fill_value=0))
    table["total"] = table.sum(axis=1)
    return events, table


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    splits = load_splits(NUM_HOSTS)
    pd.set_option("display.width", 250)
    paths = optimal_paths(splits)
    print(paths.groupby("split")[["best_return", "best_len", "safe_return",
                                  "safe_len", "delta", "optimum_needs_c1",
                                  "optimum_needs_c1_without_nontarget_reward"]]
          .mean().round(3))
    events, check = trace_m1(splits)
    print(check.to_string(index=False))
    _, table = classify(events)
    print(table)
