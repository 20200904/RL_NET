"""Generated blocks of report/reward_redesign.md (reward redesign plan).

Every number in that report is written here from csv files or from the
NASim sources and scenario files; rerun with ``python -m src.cli reward-report``.
"""
from pathlib import Path

import pandas as pd

from src.envs.generate import load_splits, make_scenario
from src.eval.readme_tables import _table, fill_blocks
from src.eval.reward_audit import CONVERGE, NUM_HOSTS, OUT, classify

REPORT = Path("report/reward_redesign.md")
NASIM_SRC = ".venv/Lib/site-packages/nasim"
SPLIT_NAMES = {"train": "training pool", "test_a": "test A", "test_b": "test B"}


def _pct(x):
    return "n/a" if pd.isna(x) else f"{100 * x:.1f}%"


def _terms():
    scenario = make_scenario(load_splits(NUM_HOSTS)["train"][0].seed, NUM_HOSTS)
    host = next(iter(scenario.hosts.values()))
    nontarget = min(float(h.value) for h in scenario.hosts.values())
    e = next(iter(scenario.exploits.values()))
    p = next(iter(scenario.privescs.values()))
    gen = f"{NASIM_SRC}/scenarios/generator.py"
    hv = f"{NASIM_SRC}/envs/host_vector.py"
    scans = " / ".join(f"−{getattr(scenario, f'{s}_scan_cost'):g}"
                       for s in ("os", "service", "subnet", "process"))
    rows = [
        ["root on a target (sensitive) host, first time",
         f"+{max(scenario.sensitive_hosts.values()):g}",
         f"`{gen}`:73-74 (`r_sensitive`, `r_user`); paid in `{hv}`:245, 284",
         "task definition (the goal)"],
        ["root on any other host, first time", f"+{nontarget:g}",
         f"`{gen}`:89 (`base_host_value`), 819-820; paid in `{hv}`:245, 284",
         "shaping (intermediate progress)"],
        ["each host newly found by a subnet scan",
         f"+{float(host.discovery_value):g}",
         f"`{gen}`:90 (`host_discovery_value`); paid in `{NASIM_SRC}/envs/network.py`:121",
         "shaping (exploration bonus)"],
        ["exploit", f"−{e['cost']:g}",
         f"`{gen}`:75 (`exploit_cost`), passed through by `src/envs/generate.py`",
         "task definition (effort and noise of an attack)"],
        ["privilege escalation", f"−{p['cost']:g}", f"`{gen}`:77 (`privesc_cost`)",
         "task definition (effort and noise of an attack)"],
        ["OS / service / subnet / process scan", scans, f"`{gen}`:79-82",
         "task definition (effort of reconnaissance)"],
        ["reward of one step", "value − cost", f"`{NASIM_SRC}/envs/environment.py`:227",
         ""],
    ]
    return _table(["term", "value now", "source", "role (our classification)"], rows)


def _priors():
    import nasim.scenarios.benchmark as bench
    from nasim.scenarios import utils as u

    def fmt(xs):
        return "/".join(f"{x:g}" for x in sorted(set(xs)))

    rows = []
    for path in sorted(Path(bench.__file__).parent.glob("*.yaml")):
        d = u.load_yaml(str(path))
        sensitive = {eval(a) if isinstance(a, str) else a for a in d[u.SENSITIVE_HOSTS]}
        other = [float(h.get(u.HOST_VALUE, u.DEFAULT_HOST_VALUE))
                 for a, h in d[u.HOST_CONFIGS].items() if eval(a) not in sensitive]
        rows.append([
            f"NASim `{path.stem}.yaml`",
            fmt(float(v) for v in d[u.SENSITIVE_HOSTS].values()),
            fmt(float(x[u.EXPLOIT_COST]) for x in d[u.EXPLOITS].values()),
            fmt(float(x[u.PRIVESC_COST]) for x in d[u.PRIVESCS].values()),
            fmt(float(d[k]) for k in (u.SERVICE_SCAN_COST, u.OS_SCAN_COST,
                                      u.SUBNET_SCAN_COST, u.PROCESS_SCAN_COST)),
            fmt(other), "0 (not set)",
        ])
    dflt = bench.generated.DEFAULTS
    rows.append(["NASim generated benchmarks (`benchmark/generated.py` DEFAULTS)",
                 f"{dflt['r_sensitive']:g}", f"{dflt['exploit_cost']:g}",
                 f"{dflt['privesc_cost']:g}", f"{dflt['service_scan_cost']:g}",
                 f"{dflt['base_host_value']:g}", f"{dflt['host_discovery_value']:g}"])
    gen = make_scenario(load_splits(NUM_HOSTS)["train"][0].seed, NUM_HOSTS)
    rows.append(["NASim `ScenarioGenerator.generate` defaults (**used here**)",
                 f"{max(gen.sensitive_hosts.values()):g}",
                 f"{next(iter(gen.exploits.values()))['cost']:g}",
                 f"{next(iter(gen.privescs.values()))['cost']:g}",
                 f"{gen.os_scan_cost:g}",
                 f"{min(float(h.value) for h in gen.hosts.values()):g}",
                 f"{float(next(iter(gen.hosts.values())).discovery_value):g}"])
    rows.append(["NASimJax (arXiv 2603.19864v3, Sec. 6.2; quoted in the text)",
                 "50", "3", "3", "1", "not stated", "1"])
    return _table(["source", "target root", "exploit cost", "privesc cost",
                   "scan cost", "other host value", "discovery"], rows)


def _problem():
    rule = pd.read_csv("results/basics/rule_agents.csv")
    rows = []
    for mode, label in (("scan", "rule agent: scan, then attack"),
                        ("scan_safe", "rule agent: scan, avoid OOS / fragile"),
                        ("blind", "rule agent: attack without scanning")):
        for split in ("train", "test_a"):
            r = rule[(rule["mode"] == mode) & (rule["split"] == split)].iloc[0]
            rows.append([label, SPLIT_NAMES[split], _pct(r["solve_rate"]),
                         f"{r['mean_steps_all']:.1f}", f"{r['mean_return']:.2f}"])
    for k in (16, 32):
        evs = pd.concat(pd.read_csv(CONVERGE / f"M1_h8_K{k}_seed{s}" / "eval.csv")
                        .assign(seed=s) for s in (0, 1, 2))
        evs = evs[evs["mode"] == "sampled"]
        for split in ("train", "test_a"):
            m = (evs[evs["split"] == split].groupby("seed")
                 [["solved", "steps", "ep_return"]].mean().mean())
            rows.append([f"M1, K={k}, 2M steps (mean of 3 seeds)",
                         SPLIT_NAMES[split], _pct(m["solved"]),
                         f"{m['steps']:.1f}", f"{m['ep_return']:.2f}"])
    return _table(["policy", "evaluated on", "solve rate", "steps / episode",
                   "return / episode"], rows)


def _delta():
    df = pd.read_csv(OUT / "audit_optimal_paths.csv")
    rows = []
    for split in ("train", "test_a", "test_b"):
        g = df[df["split"] == split]
        rows.append([
            SPLIT_NAMES[split], len(g),
            f"{g['best_return'].mean():.2f}", f"{g['safe_return'].mean():.2f}",
            f"{g['delta'].mean():.2f}", f"{g['delta'].min():g} to {g['delta'].max():g}",
            _pct((g["delta"] > 0).mean()),
            f"{g['best_len'].mean():.2f}", f"{g['safe_len'].mean():.2f}",
        ])
    return _table(["networks", "n", "best return", "best safe return", "Δ mean",
                   "Δ range", "networks with Δ > 0", "best path length",
                   "best safe path length"], rows)


C1_OUTCOMES = ("earned the +1 (root on the out-of-scope host)",
               "succeeded without the +1 (user access)", "failed attempt")


def _c1():
    df = pd.read_csv(OUT / "audit_optimal_paths.csv")
    lines = [
        f"Full-information optimum, all {len(df)} networks: the return-optimal "
        f"path needs a C1 event in {_pct(df['optimum_needs_c1'].mean())} of "
        "networks; with the non-target root reward set to 0, in "
        f"{_pct(df['optimum_needs_c1_without_nontarget_reward'].mean())}.",
        "",
        "C1 events of the converged M1 policies (2M steps, 3 seeds, sampled "
        "evaluation, 10 episodes per network):",
        "",
    ]
    _, table = classify(pd.read_csv(OUT / "audit_m1_c1_events.csv"))
    rows = []
    for (k, split), r in table.iterrows():
        rows.append([k, SPLIT_NAMES[split], int(r["total"])] + [
            f"{int(r.get(c, 0))} ({_pct(r.get(c, 0) / r['total'])})"
            for c in C1_OUTCOMES])
    lines.append(_table(["K", "evaluated on", "C1 events", "earned the +1",
                         "user access only", "failed attempt"], rows))
    return "\n".join(lines)


# ---- stage 2: candidate scoring ----

GRID_PARAMS = ("scan_cost", "attack_cost", "discovery_reward",
               "nontarget_root_reward", "target_root_reward", "failed_attack_penalty")
PARAM_HEAD = ["scan", "attack", "discovery", "non-target root", "target root",
              "failed-attack penalty"]
# options put to the user at gate 2 (the user picks; numbers are generated)
OPTIONS = {
    "1: lowest memorisation advantage, target 10":
        dict(scan_cost=0.1, attack_cost=1, discovery_reward=0, nontarget_root_reward=1,
             target_root_reward=10, failed_attack_penalty=0),
    "2: lowest memorisation advantage, target 100":
        dict(scan_cost=0.1, attack_cost=1, discovery_reward=1, nontarget_root_reward=1,
             target_root_reward=100, failed_attack_penalty=0),
    "3: prior-work ratio (NASimJax scan 1 : attack 3), target 100":
        dict(scan_cost=1, attack_cost=3, discovery_reward=1, nontarget_root_reward=0,
             target_root_reward=100, failed_attack_penalty=0),
}


def _grid():
    return pd.read_csv(OUT / "grid.csv")


def _pick(grid, params):
    mask = pd.Series(True, index=grid.index)
    for k, v in params.items():
        mask &= grid[k] == v
    assert mask.sum() == 1, params
    return grid[mask].iloc[0]


def _cand_row(r):
    return [f"{r[k]:g}" for k in GRID_PARAMS] + [
        f"{r['delta']:.2f}", _pct(r["delta_frac"]), _pct(r["delta_pos_share"]),
        f"{r['delta_test_b']:.2f}",
        f"{r['mem_adv_B']:.2f}", _pct(r["mem_adv_B_frac"]), f"{r['mem_adv_A']:.2f}",
        f"{r['scan_minus_blind']:.2f}", f"{r['B_minus_random']:.1f}",
        f"{r['robot_B_return']:.2f}", f"{r['opt_return']:.2f}",
    ]


CAND_HEAD = PARAM_HEAD + [
    "Δ", "Δ / target", "training networks with Δ > 0", "Δ test B",
    "mem. adv. (robot B)", "mem. adv. / target", "mem. adv. (robot A)",
    "robot A − blind", "robot B − random", "robot B return", "optimal return"]


def _grid_summary():
    g = _grid()
    u = g[g["is_uniform"] == 1].iloc[0]
    lines = [
        f"- Candidates scored: {len(g)}. Robot B beats random (95% bootstrap "
        f"interval over training networks above 0) in {int(g['req_B_beats_random'].sum())}; "
        f"Δ > 0 (training-pool mean) in {int(g['req_delta_pos'].sum())}; both in "
        f"{int(g['passes_required'].sum())}.",
        f"- Δ on the training pool ranges from {g['delta'].min():.2f} to "
        f"{g['delta'].max():.2f} points ({_pct(g['delta_frac'].min())} to "
        f"{_pct(g['delta_frac'].max())} of the target reward). The share of training "
        f"networks with Δ > 0 is {_pct(g['delta_pos_share'].min())} to "
        f"{_pct(g['delta_pos_share'].max())}.",
        f"- Memorisation advantage (optimal − robot B) ranges from "
        f"{g['mem_adv_B'].min():.2f} to {g['mem_adv_B'].max():.2f} points "
        f"({_pct(g['mem_adv_B_frac'].min())} to {_pct(g['mem_adv_B_frac'].max())} "
        "of the target reward). It is larger than Δ in "
        f"{int((g['mem_adv_B'] > g['delta']).sum())} of {len(g)} candidates.",
        f"- The return-optimal path needs a C1 event in at most "
        f"{_pct(g['optimum_needs_c1_share'].max())} of training networks under any candidate.",
        f"- Doing nothing useful for 100 steps (stall) never beats the optimum or robot B: "
        f"smallest margins {g['opt_minus_stall'].min():.1f} and "
        f"{g['B_minus_stall'].min():.1f} points.",
        f"- Current reward (scan 1, attack 1, discovery 1, non-target 1, target 10, no "
        f"failed-attack penalty), scored the same way: Δ {u['delta']:.2f}, memorisation "
        f"advantage {u['mem_adv_B']:.2f} ({_pct(u['mem_adv_B_frac'])} of target), robot B "
        f"return {u['robot_B_return']:.2f} against an optimal {u['opt_return']:.2f}, "
        f"robot A − blind {u['scan_minus_blind']:.2f}.",
    ]
    return "\n".join(lines)


def _grid_axes():
    g = _grid()
    rows = []
    for k, head in zip(GRID_PARAMS, PARAM_HEAD):
        for v, s in g.groupby(k):
            rows.append([head, f"{v:g}", f"{s['delta'].mean():.2f}",
                         _pct(s["delta_frac"].mean()), _pct(s["delta_pos_share"].mean()),
                         f"{s['mem_adv_B'].mean():.2f}", _pct(s["mem_adv_B_frac"].mean()),
                         f"{s['scan_minus_blind'].mean():.2f}"])
    return _table(["axis", "value", "Δ", "Δ / target", "training networks with Δ > 0",
                   "mem. adv. (robot B)", "mem. adv. / target", "robot A − blind"], rows)


def _grid_top(target):
    def block():
        g = _grid()
        s = (g[(g["target_root_reward"] == target) & (g["passes_required"] == 1)]
             .sort_values(["mem_adv_B", "delta"], ascending=[True, False]).head(8))
        return _table(CAND_HEAD, [_cand_row(r) for _, r in s.iterrows()])
    return block


def _grid_options():
    g = _grid()
    rows = []
    for label, params in OPTIONS.items():
        for pen in (0, 2):
            r = _pick(g, {**params, "failed_attack_penalty": pen})
            rows.append([label if pen == 0 else f"{label.split(':')[0]} + failed-attack penalty 2"]
                        + _cand_row(r))
    u = g[g["is_uniform"] == 1].iloc[0]
    rows.append(["current reward"] + _cand_row(u))
    return _table(["option"] + CAND_HEAD, rows)


def _grid_behaviour():
    e = pd.read_csv(OUT / "grid_episodes.csv")
    e = e[e["split"] == "train"].copy()
    e["group"] = e["policy"].str.replace(r"_seed\d+$", "", regex=True)
    labels = {"rule_scan_safe": "robot B (scan, avoid OOS / fragile)",
              "rule_scan": "robot A (scan, then attack)",
              "rule_blind": "attack without scanning", "random": "random",
              "M1_K16": "M1, K=16 (3 seeds)", "M1_K32": "M1, K=32 (3 seeds)"}
    paths = pd.read_csv(OUT / "grid_paths.csv")
    rows = []
    for group, label in labels.items():
        s = e[e["group"] == group]
        rows.append([label, _pct(s["solved"].mean()), f"{s['steps'].mean():.1f}",
                     f"{s['u_attack'].mean():.1f}", f"{s['u_scan'].mean():.1f}",
                     f"{s['u_failed'].mean():.2f}", f"{s['u_discovered'].mean():.1f}"])
    u = paths[(paths["split"] == "train")]
    rows.append(["optimal path (over all candidates)", "100.0%",
                 f"{u['opt_len'].min():g} to {u['opt_len'].max():g}", "", "", "0", ""])
    return _table(["policy (training pool)", "solve rate", "steps / episode",
                   "attacks", "scans", "failed attacks", "hosts discovered"], rows)


BLOCKS = {
    "terms": _terms,
    "priors": _priors,
    "problem": _problem,
    "delta": _delta,
    "c1": _c1,
    "grid_summary": _grid_summary,
    "grid_behaviour": _grid_behaviour,
    "grid_axes": _grid_axes,
    "grid_top_10": _grid_top(10.0),
    "grid_top_100": _grid_top(100.0),
    "grid_options": _grid_options,
}


def run():
    fill_blocks(REPORT, BLOCKS)
    print(f"{REPORT} regenerated")
