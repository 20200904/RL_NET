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
CONVERGE = {"1M": PHASE1 / "converge" / "1m", "2M": PHASE1 / "converge" / "2m"}
SPLIT_NAMES = {"train": "training pool", "test_a": "test A (held-out)",
               "test_b": "test B (more fragile hosts)"}
SPLIT_ORDER = {"train": 0, "test_a": 1, "test_b": 2}


def _pct(x):
    return "n/a" if pd.isna(x) else f"{100 * x:.1f}%"


def _num(x):
    return "n/a" if pd.isna(x) else f"{x:.2f}"


def _range(row, metric, fmt=_pct):
    return (f"{fmt(row[metric + '_mean'])} "
            f"({fmt(row[metric + '_min'])}–{fmt(row[metric + '_max'])})")


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


RESULT_HEADERS = [
    "method", "hosts", "K", "train steps", "evaluated on", "seeds",
    "solve rate, mean (min–max)", "violations among solved, mean (min–max)",
    "safe solve rate", "C1 / episode", "C2 / episode",
    "cost per attack attempt", "scan before exploit", "steps when solved",
]


def _result_rows(exp, steps_label):
    df = pd.read_csv(exp / "summary.csv")
    df = df[df["mode"] == "sampled"].copy()
    df["order"] = df["split"].map(SPLIT_ORDER)
    rows = []
    for _, r in df.sort_values(["num_hosts", "method", "k", "order"]).iterrows():
        rows.append([
            r["method"], r["num_hosts"], r["k"],
            "none" if r["method"] == "M0" else steps_label,
            SPLIT_NAMES[r["split"]], r["seeds"],
            _range(r, "solve_rate"), _range(r, "violation_rate_among_solved"),
            _pct(r["safe_solve_rate_mean"]),
            _num(r["cost_c1_per_episode_mean"]),
            _num(r["cost_c2_per_episode_mean"]),
            _num(r["cost_per_attack_attempt_mean"]),
            _pct(r["scan_before_exploit_mean"]),
            _num(r["mean_steps_solved_mean"]),
        ])
    return rows


def _k4():
    return _table(RESULT_HEADERS, _result_rows(PHASE1 / "runs", "300k"))


def _converged():
    rows = []
    for label, exp in CONVERGE.items():
        rows += _result_rows(exp, label)
    return _table(RESULT_HEADERS, rows)


def _steps_to_converge():
    df = pd.read_csv(CONVERGE["2M"] / "steps_to_converge.csv")
    rows = [[
        r["k"], r["seed"],
        "not reached" if pd.isna(r["first_step_solve_ge_90"])
        else f'{int(r["first_step_solve_ge_90"]):,}',
        "not reached" if pd.isna(r["first_step_flat"])
        else f'{int(r["first_step_flat"]):,}',
        _pct(r["final_solve_rate"]), _num(r["final_ep_len"]),
        _num(r["final_cost"]),
    ] for _, r in df.sort_values(["k", "seed"]).iterrows()]
    return _table(
        ["K", "seed", "first step with train solve rate ≥ 90%",
         "first step where it is flat", "solve rate, last 200k steps",
         "episode length, last 200k", "cost / episode, last 200k"], rows)


GAP_COLUMNS = [
    ("solve_rate", "solve rate", _pct),
    ("violation_rate_among_solved", "violations among solved", _pct),
    ("cost_c1_per_episode", "C1 / episode", _num),
    ("cost_c2_per_episode", "C2 / episode", _num),
    ("cost_per_attack_attempt", "cost per attack attempt", _num),
]


def _gaps():
    rows = []
    for label, exp in CONVERGE.items():
        df = pd.read_csv(exp / "gaps.csv")
        df = df[(df["mode"] == "sampled") & (df["rq3_included"] == 1)]
        for (k, test), part in df.groupby(["k", "test_split"]):
            part = part.set_index("metric")
            row = [k, label, SPLIT_NAMES[test], int(part["seeds"].iloc[0])]
            for metric, _, fmt in GAP_COLUMNS:
                r = part.loc[metric]
                sign = "+" if r["gap_mean"] >= 0 else "−"
                # a difference of two rates is in percentage points
                gap = (f'{100 * abs(r["gap_mean"]):.1f} pp' if fmt is _pct
                       else fmt(abs(r["gap_mean"])))
                row.append(f'{fmt(r["train_mean"])} → {fmt(r["test_mean"])} '
                           f'({sign}{gap})')
            rows.append(row)
    headers = ["K", "train steps", "test set", "seeds"] + [
        f"{name}: train → test (gap)" for _, name, _ in GAP_COLUMNS]
    return _table(headers, rows)


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
    "k4": _k4,
    "steps_to_converge": _steps_to_converge,
    "converged": _converged,
    "gaps": _gaps,
    "ppo_validation": _ppo_validation,
}


def fill_blocks(path, blocks):
    """Rewrite every generated block of ``path``; a missing csv gives "not run"."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    for name, build in blocks.items():
        try:
            body = build()
        except FileNotFoundError:
            body = "not run"
        pattern = re.compile(
            rf"(<!-- BEGIN GENERATED: {name} -->).*?(<!-- END GENERATED: {name} -->)",
            re.S)
        if not pattern.search(text):
            print(f"marker for '{name}' not found in {path}")
            continue
        text = pattern.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(2)}", text)
    path.write_text(text, encoding="utf-8")


def run():
    fill_blocks(README, BLOCKS)
    print("README.md tables regenerated")
