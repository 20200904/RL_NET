"""Reward redesign, stage 2: score reward candidates (no training).

A candidate reward is a linear function of six per-step features:

    r = -attack_cost * attack - scan_cost * scan + discovery_reward * discovered
        + target_root_reward * root_target + nontarget_root_reward * root_nontarget
        - failed_attack_penalty * failed

- attack:         1 for an exploit or privilege escalation
- scan:           1 for an OS, service, process or subnet scan (one
                  reconnaissance cost for all four)
- discovered:     hosts newly found by a successful subnet scan
- root_target:    1 the first time root is reached on a target (sensitive) host
- root_nontarget: 1 the first time root is reached on any other host
- failed:         1 for an exploit or privilege escalation after which the
                  agent's access level on the target host is not higher than
                  before (no new access); includes connection and permission
                  errors and attacks on hosts already held at that level

So the gamma-discounted return of a recorded episode under any candidate is
the candidate's coefficient vector times the episode's discounted feature
sums. Each policy is rolled out once and every candidate is scored from the
recorded sums.

Optimal and optimal-safe paths are not taken from the earlier shortest-path
search: with unequal costs and discounting the shortest path need not have
the highest return. For each network a depth-first search over NASim states
(full information) computes, for all candidates at once,

    V(s) = max over state-changing actions a of  r(s, a) + gamma * V(s')

with V = r at the goal. Every action changes the state only by adding access
or discovered hosts, so the state graph has no cycles and this recursion is
exact. Actions that do not change the state (OS / service / process scans,
failed or repeated attacks) only add a cost and delay, so with full
information they are never on an optimal path; consequently the failed-attack
penalty cannot change the optimal or optimal-safe return (checked below).
Ties are broken towards the shorter path.
"""
import itertools
import json
import multiprocessing as mp
import time

import numpy as np
import pandas as pd
import torch

from src.agents.ppo import Agent
from src.agents.random_agent import RandomAgent
from src.agents.rule_agent import RuleAgent
from src.envs.generate import load_splits, make_env
from src.envs.multi_env import MultiNetEnv
from src.eval.reward_audit import (CONVERGE, NUM_HOSTS, OUT, STEP_LIMIT,
                                   PathSearch, _no_c1, _no_cost)

GAMMA = 0.99                  # PPOConfig.gamma, what PPO maximises
FEATURES = ("attack", "scan", "discovered", "root_target", "root_nontarget", "failed")
AXES = dict(
    scan_cost=(0.1, 0.25, 0.5, 1.0),
    attack_cost=(1.0, 2.0, 3.0),
    discovery_reward=(0.0, 1.0),
    nontarget_root_reward=(0.0, 1.0),
    target_root_reward=(10.0, 100.0),
    failed_attack_penalty=(0.0, 2.0),
)
PARAMS = tuple(AXES)
UNIFORM = dict(scan_cost=1.0, attack_cost=1.0, discovery_reward=1.0,
               nontarget_root_reward=1.0, target_root_reward=10.0,
               failed_attack_penalty=0.0)
EVAL_EPISODES = 10            # per network, as in the M1 evaluation
RANDOM_SEED = 0
BOOT = 2000                   # bootstrap resamples over networks


def candidates():
    df = pd.DataFrame(list(itertools.product(*AXES.values())), columns=PARAMS)
    df.insert(0, "candidate", range(len(df)))
    return df


def coefficients(params):
    """(n, 6) coefficient matrix for a frame (or dict) of reward parameters."""
    p = pd.DataFrame([params]) if isinstance(params, dict) else params
    return np.stack([-p["attack_cost"], -p["scan_cost"], p["discovery_reward"],
                     p["target_root_reward"], p["nontarget_root_reward"],
                     -p["failed_attack_penalty"]], axis=1).astype(float)


def step_features(action, result, sensitive, new_access):
    f = np.zeros(len(FEATURES))
    attack = action.is_exploit() or action.is_privilege_escalation()
    f[0 if attack else 1] = 1.0
    if action.is_subnet_scan() and result["success"]:
        f[2] = sum(result["newly_discovered"].values())
    elif attack and result["success"] and result["value"] > 0:
        f[3 if tuple(action.target) in sensitive else 4] = 1.0
    if attack and not new_access:
        f[5] = 1.0
    return f


def _new_access(before, after, action):
    target = tuple(action.target)
    return after.get_host(target).access > before.get_host(target).access


# ---- recorded episodes ----

def record(agent, specs, split, episodes, deterministic, seed, policy):
    """Roll ``agent`` out exactly as ``run_eval.evaluate`` does (same seeding)
    and return one row per episode with discounted and undiscounted feature
    sums."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = MultiNetEnv(specs, STEP_LIMIT)
    uniform = coefficients(UNIFORM)[0]
    rows = []
    for net_idx, spec in enumerate(specs):
        sensitive = set(env._scenarios[net_idx].sensitive_addresses)
        for episode in range(episodes):
            obs, _ = env.reset(options={"network": net_idx})
            if hasattr(agent, "reset"):
                agent.reset()
            nasim = env._env.unwrapped
            disc, undisc = np.zeros(len(FEATURES)), np.zeros(len(FEATURES))
            t, ret, c1, c2 = 0, 0.0, 0, 0
            while True:
                a = int(agent.act(obs, deterministic))
                action = nasim.action_space.get_action(a)
                before = nasim.current_state
                obs, r, term, trunc, info = env.step(a)
                new = _new_access(before, nasim.current_state, action)
                f = step_features(action, info, sensitive, new)
                if f[3] or f[4]:
                    assert new, "a root reward without new access"
                disc += GAMMA ** t * f
                undisc += f
                t += 1
                ret += float(r)
                c1 += info["cost_c1"]
                c2 += info["cost_c2"]
                if term or trunc:
                    break
            if not np.isclose(uniform @ undisc, ret):
                raise AssertionError(f"recomputed return {uniform @ undisc} != {ret}")
            row = dict(policy=policy, split=split, network_seed=spec.seed,
                       episode=episode, solved=int(term), steps=t, ep_return=ret,
                       cost_c1=c1, cost_c2=c2)
            row.update({f"d_{k}": v for k, v in zip(FEATURES, disc)})
            row.update({f"u_{k}": v for k, v in zip(FEATURES, undisc)})
            rows.append(row)
    return rows


def _record_job(job):
    kind, arg = job
    splits = load_splits(NUM_HOSTS)
    rows = []
    if kind == "rule":
        agent = RuleAgent(make_env(splits["train"][0], STEP_LIMIT), arg)
        for split, specs in splits.items():
            # agent and environment are deterministic: one episode per network
            rows += record(agent, specs, split, 1, True, 0, f"rule_{arg}")
    elif kind == "random":
        probe = MultiNetEnv(splits["train"][:1], STEP_LIMIT)
        agent = RandomAgent(int(probe.action_space.n), seed=RANDOM_SEED)
        for split, specs in splits.items():
            rows += record(agent, specs, split, EVAL_EPISODES, False,
                           RANDOM_SEED, "random")
    else:
        k, seed = arg
        run_dir = CONVERGE / f"M1_h8_K{k}_seed{seed}"
        meta = json.loads((run_dir / "done.json").read_text())
        probe = MultiNetEnv(splits["train"][:1], STEP_LIMIT)
        agent = Agent(probe.observation_space.shape[0], int(probe.action_space.n),
                      meta["ppo"]["hidden"])
        agent.load_state_dict(torch.load(run_dir / "model.pt"))
        ev = pd.read_csv(run_dir / "eval.csv")
        sets = dict(train=splits["train"][:k], test_a=splits["test_a"],
                    test_b=splits["test_b"])
        for split, specs in sets.items():
            got = record(agent, specs, split, meta["eval_episodes"], False, seed,
                         f"M1_K{k}_seed{seed}")
            want = ev[(ev["mode"] == "sampled") & (ev["split"] == split)]
            g = pd.DataFrame(got)
            for col in ("solved", "steps", "cost_c1", "cost_c2"):
                if not np.array_equal(g[col].to_numpy(), want[col].to_numpy()):
                    raise AssertionError(f"{run_dir} {split}: replay differs in {col}")
            if not np.allclose(g["ep_return"], want["ep_return"]):
                raise AssertionError(f"{run_dir} {split}: replay differs in return")
            rows += got
    return rows


# ---- optimal paths, all candidates at once ----

def best_paths(search, coef, gamma, allow):
    """Best discounted return to the goal and its length, per candidate."""
    n = coef.shape[0]
    memo, feats = {}, {}

    def go(state, key):
        if key in memo:
            return memo[key]
        bv, bl = np.full(n, -np.inf), np.full(n, np.inf)
        for i, (a, nxt, nkey, done, _, result) in enumerate(search._edges_from(state)):
            if not allow(*search.cost[id(a)]):
                continue
            if (key, i) not in feats:
                feats[(key, i)] = step_features(a, result, search.sensitive,
                                                _new_access(state, nxt, a))
            f = feats[(key, i)]
            assert f[5] == 0, "a state-changing attack without new access"
            r = coef @ f
            if done:
                v, length = r, 1.0
            else:
                sv, sl = go(nxt, nkey)
                if not np.isfinite(sv[0]):
                    continue
                v, length = r + gamma * sv, 1.0 + sl
            better = (v > bv + 1e-9) | ((np.abs(v - bv) <= 1e-9) & (length < bl))
            bv, bl = np.where(better, v, bv), np.where(better, length, bl)
        memo[key] = (bv, bl)
        return memo[key]

    return go(search.start, search.start.tensor.tobytes())


def _paths_job(item):
    split, net = item
    torch.set_num_threads(1)
    cands = candidates()
    coef = coefficients(cands)
    search = PathSearch(net)
    best, best_len = best_paths(search, coef, GAMMA, lambda c1, c2: True)
    safe, safe_len = best_paths(search, coef, GAMMA, _no_cost)
    no_c1, _ = best_paths(search, coef, GAMMA, _no_c1)
    # stage-1 check: uniform reward, no discount, equals audit_optimal_paths.csv
    u = coefficients(UNIFORM)
    ub, _ = best_paths(search, u, 1.0, lambda c1, c2: True)
    us, _ = best_paths(search, u, 1.0, _no_cost)
    return pd.DataFrame(dict(
        split=split, network_seed=net.seed, candidate=cands["candidate"],
        opt_return=best, opt_len=best_len, safe_return=safe, safe_len=safe_len,
        optimum_needs_c1=(no_c1 < best - 1e-9).astype(int),
        uniform_undiscounted_best=ub[0], uniform_undiscounted_safe=us[0],
    ))


# ---- scoring ----

def stall_return(scan_cost, attack_cost):
    """Return of spending the whole step limit on the cheapest action without
    ever reaching the goal (no gains): a floor that any policy can secure."""
    return -np.minimum(scan_cost, attack_cost) * sum(GAMMA ** t for t in range(STEP_LIMIT))


def score(episodes, paths, cands):
    coef = coefficients(cands)                                   # (C, 6)
    dcols = [f"d_{k}" for k in FEATURES]
    ep = episodes.copy()
    ret = ep[dcols].to_numpy() @ coef.T                          # (E, C)
    # per policy group: M1 seeds are averaged per K
    ep["group"] = ep["policy"].str.replace(r"_seed\d+$", "", regex=True)

    def per_network(group, split):
        mask = ((ep["group"] == group) & (ep["split"] == split)).to_numpy()
        frame = pd.DataFrame(ret[mask]).assign(net=ep.loc[mask, "network_seed"].to_numpy())
        return frame.groupby("net").mean()                       # (networks, C)

    long_rows = []
    for (group, split), _ in ep.groupby(["group", "split"]):
        pn = per_network(group, split)
        for c, v in enumerate(pn.mean().to_numpy()):
            long_rows.append(dict(candidate=c, split=split, policy=group, mean_return=v))
    for split, g in paths.groupby("split"):
        means = g.groupby("candidate")[["opt_return", "safe_return"]].mean()
        for c in means.index:
            long_rows.append(dict(candidate=c, split=split, policy="optimal",
                                  mean_return=means.loc[c, "opt_return"]))
            long_rows.append(dict(candidate=c, split=split, policy="optimal_safe",
                                  mean_return=means.loc[c, "safe_return"]))
    by_split = pd.DataFrame(long_rows)

    def pol(policy, split="train"):
        s = by_split[(by_split["policy"] == policy) & (by_split["split"] == split)]
        return s.set_index("candidate")["mean_return"].reindex(cands["candidate"]).to_numpy()

    out = cands.copy()
    tgt = out["target_root_reward"].to_numpy()
    for split, tag in (("train", ""), ("test_a", "_test_a"), ("test_b", "_test_b")):
        g = paths[paths["split"] == split].copy()
        g["delta"] = g["opt_return"] - g["safe_return"]
        agg = g.groupby("candidate").agg(
            opt_return=("opt_return", "mean"), opt_len=("opt_len", "mean"),
            safe_return=("safe_return", "mean"), safe_len=("safe_len", "mean"),
            delta=("delta", "mean"), delta_pos_share=("delta", lambda d: (d > 1e-9).mean()),
            optimum_needs_c1_share=("optimum_needs_c1", "mean"),
        ).reindex(out["candidate"])
        keep = (agg.columns if split == "train"
                else ["delta", "delta_pos_share"])
        for col in keep:
            out[col + tag] = agg[col].to_numpy()
        out["delta_frac" + tag] = out["delta" + tag] / tgt

    names = dict(rule_scan_safe="robot_B_return", rule_scan="robot_A_return",
                 rule_blind="blind_return", random="random_return",
                 M1_K16="m1_k16_train_return", M1_K32="m1_k32_train_return")
    for policy, col in names.items():
        out[col] = pol(policy)
    out["m1_k16_test_a_return"] = pol("M1_K16", "test_a")
    out["m1_k32_test_a_return"] = pol("M1_K32", "test_a")
    out["robot_B_test_a_return"] = pol("rule_scan_safe", "test_a")

    out["mem_adv_B"] = out["opt_return"] - out["robot_B_return"]
    out["mem_adv_B_frac"] = out["mem_adv_B"] / tgt
    out["mem_adv_A"] = out["opt_return"] - out["robot_A_return"]
    out["scan_minus_blind"] = out["robot_A_return"] - out["blind_return"]
    out["B_minus_random"] = out["robot_B_return"] - out["random_return"]

    # paired bootstrap over training networks of (robot B - random)
    b = per_network("rule_scan_safe", "train")
    r = per_network("random", "train").reindex(b.index)
    diff = (b - r).to_numpy()                                    # (N, C)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, diff.shape[0], size=(BOOT, diff.shape[0]))
    boot = diff[idx].mean(axis=1)                                # (BOOT, C)
    out["B_minus_random_ci_low"] = np.quantile(boot, 0.025, axis=0)
    out["B_minus_random_ci_high"] = np.quantile(boot, 0.975, axis=0)

    out["stall_return"] = stall_return(out["scan_cost"], out["attack_cost"])
    out["opt_minus_stall"] = out["opt_return"] - out["stall_return"]
    out["B_minus_stall"] = out["robot_B_return"] - out["stall_return"]
    out["return_scale"] = out["opt_return"] - out["random_return"]
    out["req_B_beats_random"] = (out["B_minus_random_ci_low"] > 0).astype(int)
    out["req_delta_pos"] = (out["delta"] > 1e-9).astype(int)
    out["passes_required"] = out["req_B_beats_random"] * out["req_delta_pos"]
    out["is_uniform"] = np.all([out[k] == v for k, v in UNIFORM.items()], axis=0).astype(int)
    return out, by_split


def pareto_front(df, x="delta", y="mem_adv_B", decimals=2):
    """Indices of candidates not dominated in (maximise x, minimise y),
    compared at reporting precision (``decimals``) so that differences in
    the fourth decimal do not keep a dominated candidate on the front."""
    keep = []
    xs, ys = df[x].round(decimals), df[y].round(decimals)
    for i in df.index:
        dominated = ((xs >= xs[i]) & (ys <= ys[i])
                     & ((xs > xs[i]) | (ys < ys[i]))).any()
        if not dominated:
            keep.append(i)
    return keep


def plot(grid, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # categorical slots 1-2 of the dataviz reference palette; marker shape is
    # a second encoding so identity does not rest on colour alone
    colors = {10.0: "#2a78d6", 100.0: "#eb6834"}
    markers = {10.0: "o", 100.0: "^"}
    plt.rcParams.update({"axes.edgecolor": "#8a8a86", "axes.labelcolor": "#33332f",
                         "xtick.color": "#5c5c58", "ytick.color": "#5c5c58",
                         "text.color": "#33332f"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#fcfcfb")
    for ax, (x, y, unit, dec) in zip(axes, (("delta", "mem_adv_B", "points", 2),
                                            ("delta_frac", "mem_adv_B_frac",
                                             "÷ target reward", 4))):
        for tgt, g in grid.groupby("target_root_reward"):
            ok = g[g["passes_required"] == 1]
            bad = g[g["passes_required"] == 0]
            ax.scatter(ok[x], ok[y], s=64, marker=markers[tgt], color=colors[tgt],
                       alpha=0.8, edgecolors="#fcfcfb", linewidths=1.5,
                       label=f"target {tgt:g}")
            if len(bad):
                ax.scatter(bad[x], bad[y], s=64, marker=markers[tgt],
                           facecolors="none", edgecolors=colors[tgt],
                           linewidths=1.2, alpha=0.7,
                           label=f"target {tgt:g}, fails a requirement")
            if len(ok):
                front = ok.loc[pareto_front(ok, x, y, dec)].sort_values(x)
                ax.plot(front[x], front[y], color=colors[tgt], lw=2)
        ax.set_facecolor("#fcfcfb")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        u = grid[grid["is_uniform"] == 1]
        ax.scatter(u[x], u[y], marker="*", s=220, color="#33332f", zorder=5,
                   edgecolors="#fcfcfb", linewidths=1.5,
                   label="current reward (uniform)")
        ax.set_xlabel(f"Δ = optimal − optimal-safe return ({unit})")
        ax.set_ylabel("memorisation advantage = optimal − robot B\n"
                      f"({unit})")
        ax.grid(alpha=0.25, color="#c8c8c4", lw=0.8)
    axes[0].legend(fontsize=8)
    fig.suptitle("Reward candidates, training pool, γ = 0.99 "
                 "(lines: Pareto front among candidates meeting the requirements)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def check_paths(paths):
    """Uniform reward without discount must reproduce stage 1, and the
    failed-attack penalty must not change any optimal or optimal-safe return."""
    audit = pd.read_csv(OUT / "audit_optimal_paths.csv")
    chk = (paths.drop_duplicates("network_seed")
           [["split", "network_seed", "uniform_undiscounted_best",
             "uniform_undiscounted_safe"]]
           .merge(audit, on=["split", "network_seed"]))
    assert len(chk) == len(audit)
    assert np.allclose(chk["uniform_undiscounted_best"], chk["best_return"])
    assert np.allclose(chk["uniform_undiscounted_safe"], chk["safe_return"])
    p = paths.merge(candidates(), on="candidate")
    key = [k for k in PARAMS if k != "failed_attack_penalty"] + ["network_seed"]
    for col in ("opt_return", "safe_return"):
        assert p.groupby(key)[col].nunique().max() == 1
    assert np.isfinite(paths[["opt_return", "safe_return"]].to_numpy()).all()
    assert paths["opt_len"].max() <= STEP_LIMIT and paths["safe_len"].max() <= STEP_LIMIT


def rescore():
    """Recompute grid.csv and the plot from the saved episodes and paths."""
    episodes = pd.read_csv(OUT / "grid_episodes.csv")
    paths = pd.read_csv(OUT / "grid_paths.csv")
    check_paths(paths)
    grid, by_split = score(episodes, paths, candidates())
    grid.to_csv(OUT / "grid.csv", index=False)
    by_split.to_csv(OUT / "grid_by_split.csv", index=False)
    plot(grid, OUT / "grid_pareto.png")
    return grid


def run(workers=3):
    OUT.mkdir(parents=True, exist_ok=True)
    splits = load_splits(NUM_HOSTS)
    t0 = time.time()
    jobs = ([("M1", (k, s)) for k in (16, 32) for s in (0, 1, 2)]
            + [("rule", m) for m in ("scan_safe", "scan", "blind")]
            + [("random", None)])
    items = [(split, net) for split, specs in splits.items() for net in specs]
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=1) as pool:
        ep_async = pool.map_async(_record_job, jobs, chunksize=1)
        paths = pd.concat(pool.map(_paths_job, items, chunksize=1), ignore_index=True)
        episodes = pd.DataFrame([r for rows in ep_async.get() for r in rows])
    print(f"rollouts and path search: {time.time() - t0:.0f} s", flush=True)
    episodes.to_csv(OUT / "grid_episodes.csv", index=False)
    paths.to_csv(OUT / "grid_paths.csv", index=False)

    check_paths(paths)
    c = candidates()
    grid, by_split = score(episodes, paths, c)
    grid.to_csv(OUT / "grid.csv", index=False)
    by_split.to_csv(OUT / "grid_by_split.csv", index=False)
    plot(grid, OUT / "grid_pareto.png")
    print(f"wrote {OUT / 'grid.csv'} ({len(grid)} candidates)")
    return grid
