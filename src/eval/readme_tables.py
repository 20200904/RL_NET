"""Regenerate the result tables in README.md from csv files under results/.

Every number in the README lives between a pair of markers

    <!-- BEGIN GENERATED: name -->  ...  <!-- END GENERATED: name -->

and is rewritten by this script. A table whose csv is missing says "not run".
"""
import re
from pathlib import Path

import pandas as pd

README = Path("README.md")
PHASE1 = Path("results/phase1")
SPLIT_NAMES = {"train": "training pool", "test_a": "test A (held-out)",
               "test_b": "test B (more fragile hosts)"}


def _pct(x):
    return "n/a" if pd.isna(x) else f"{100 * x:.1f}%"


def _num(x):
    return "n/a" if pd.isna(x) else f"{x:.2f}"


def _range(row, metric):
    return (f"{_pct(row[metric + '_mean'])} "
            f"({_pct(row[metric + '_min'])}–{_pct(row[metric + '_max'])})")


def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _calibration():
    rows = []
    for num_hosts in (6, 8):
        cal = pd.read_csv(PHASE1 / f"calibration_h{num_hosts}.csv")
        paths = pd.read_csv(PHASE1 / f"path_lengths_h{num_hosts}.csv")
        train = paths[paths["split"] == "train"]
        at = lambda limit: _pct(
            cal.loc[cal["step_limit"] == limit, "random_solve_rate"].iloc[0])
        rows.append([
            num_hosts, at(50), at(100), at(200),
            f'{train["safe_path_len"].min()}–{train["safe_path_len"].max()}',
            *[_pct(paths.loc[paths["split"] == s, "shortcut"].mean())
              for s in ("train", "test_a", "test_b")],
        ])
    return _table(
        ["hosts", "random solve @50 steps", "@100", "@200",
         "shortest zero-cost path (train)", "shortcut networks: train",
         "test A", "test B"], rows)


def _results(exp, label_k=False):
    df = pd.read_csv(exp / "summary.csv")
    df = df[df["mode"] == "sampled"]
    rows = []
    for _, r in df.sort_values(["num_hosts", "method", "k", "split"],
                               key=lambda s: s.map(
                                   {"train": 0, "test_a": 1, "test_b": 2}
                               ) if s.name == "split" else s).iterrows():
        rows.append([
            r["method"], r["num_hosts"], r["k"], SPLIT_NAMES[r["split"]],
            r["seeds"], _range(r, "solve_rate"), _range(r, "violation_rate"),
            _pct(r["violation_rate_among_solved_mean"]),
            _pct(r["safe_solve_rate_mean"]),
            _num(r["mean_cost_c1_mean"]), _num(r["mean_cost_c2_mean"]),
            _num(r["mean_steps_solved_mean"]),
        ])
    return _table(
        ["method", "hosts", "K", "evaluated on", "seeds",
         "solve rate, mean (min–max)", "violation rate, mean (min–max)",
         "violations among solved", "safe solve rate", "C1 / episode",
         "C2 / episode", "steps when solved"], rows)


def _gaps():
    df = pd.read_csv(PHASE1 / "runs" / "gaps.csv")
    df = df[(df["mode"] == "sampled") & (df["method"] == "M1")]
    rows = [[
        r["num_hosts"], r["k"], SPLIT_NAMES[r["test_split"]], r["seeds"],
        _pct(r["train_solve_rate"]), _pct(r["test_solve_rate"]),
        _pct(r["capability_gap"]), _pct(r["train_violation_rate"]),
        _pct(r["test_violation_rate"]), _pct(r["safety_gap"]),
    ] for _, r in df.iterrows()]
    return _table(
        ["hosts", "K", "test set", "seeds", "solve: train", "solve: test",
         "capability gap", "violation: train", "violation: test",
         "safety gap"], rows)


def _ppo_validation():
    df = pd.read_csv(PHASE1 / "validate_ppo" / "summary.csv")
    rows = [[
        "hand-written" if r["impl"] == "own" else "SB3", r["seed"],
        r["train_steps"], _num(r["last100_train_ep_len"]),
        _pct(r["eval_sampled_solve_rate"]), f'{r["steps_per_sec"]:.0f}',
    ] for _, r in df.sort_values(["impl", "seed"]).iterrows()]
    return _table(
        ["PPO", "seed", "train steps", "episode length, last 100 train episodes",
         "sampled solve rate", "steps / s"], rows)


BLOCKS = {
    "calibration": _calibration,
    "rq1": lambda: _results(PHASE1 / "runs"),
    "gaps": _gaps,
    "probe": lambda: _results(PHASE1 / "probe_k"),
    "ppo_validation": _ppo_validation,
}


def run():
    text = README.read_text(encoding="utf-8")
    for name, build in BLOCKS.items():
        try:
            body = build()
        except FileNotFoundError:
            body = "not run"
        pattern = re.compile(
            rf"(<!-- BEGIN GENERATED: {name} -->).*?(<!-- END GENERATED: {name} -->)",
            re.S)
        if not pattern.search(text):
            print(f"marker for '{name}' not found in README.md")
            continue
        text = pattern.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(2)}", text)
    README.write_text(text, encoding="utf-8")
    print("README.md tables regenerated")
