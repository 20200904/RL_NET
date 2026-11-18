"""Summarise the eval.csv files of an experiment directory.

Writes next to the runs:
  summary_runs.csv  one row per (run, split, mode)
  summary.csv       mean / min / max over seeds
  gaps_runs.csv     train-vs-test gap per run, test set and metric (long format)
  gaps.csv          the same, mean / min / max over seeds

With few seeds only mean / min / max are reported. IQM, stratified bootstrap
CIs and probability of improvement (rliable) are added for the tier runs.
"""
import json
from pathlib import Path

import pandas as pd

KEYS = ["method", "num_hosts", "k"]
METRICS = [
    "solve_rate", "violation_rate", "violation_rate_among_solved",
    "safe_solve_rate", "cost_c1_per_episode", "cost_c2_per_episode",
    "cost_per_attack_attempt", "mean_steps_solved", "scan_before_exploit",
    "c2_unscanned_frac",
]
# Metrics a generalization gap is computed for. The sign makes a positive
# gap mean "worse on the unseen networks": train - test for the solve rate,
# test - train for everything that counts harm.
GAP_METRICS = {
    "solve_rate": "train_minus_test",
    "cost_c1_per_episode": "test_minus_train",
    "cost_c2_per_episode": "test_minus_train",
    "cost_per_attack_attempt": "test_minus_train",
    "violation_rate_among_solved": "test_minus_train",
    "violation_rate": "test_minus_train",
}
# K=4 policies memorise their training networks (Phase 1), so their gaps do
# not answer RQ3. They are kept as a baseline and flagged, see DECISIONS.md.
RQ3_MIN_K = 16


def _ratio(num, den):
    return num / den if den else float("nan")


def summarise_episodes(eps):
    solved = eps[eps["solved"] == 1]
    return dict(
        episodes=len(eps),
        solve_rate=eps["solved"].mean(),
        violation_rate=(eps["cost"] > 0).mean(),
        # violation rate restricted to goal-reaching episodes
        violation_rate_among_solved=(
            (solved["cost"] > 0).mean() if len(solved) else float("nan")
        ),
        safe_solve_rate=((eps["solved"] == 1) & (eps["cost"] == 0)).mean(),
        cost_c1_per_episode=eps["cost_c1"].mean(),
        cost_c2_per_episode=eps["cost_c2"].mean(),
        # cost events per attack attempt (exploit or privilege escalation,
        # the only actions that can incur a cost)
        cost_per_attack_attempt=_ratio(
            eps["cost"].sum(), eps["exploits"].sum() + eps["privescs"].sum()),
        mean_steps_solved=solved["steps"].mean() if len(solved) else float("nan"),
        scan_before_exploit=_ratio(eps["exploits_scanned"].sum(),
                                   eps["exploits"].sum()),
        c2_unscanned_frac=_ratio(eps["c2_unscanned"].sum(), eps["cost_c2"].sum()),
    )


def _gaps(runs):
    index = KEYS + ["seed", "mode"]
    train = runs[runs["split"] == "train"].set_index(index)
    rows = []
    for test in ("test_a", "test_b"):
        test_rows = runs[runs["split"] == test].set_index(index)
        for metric, direction in GAP_METRICS.items():
            for key, test_value in test_rows[metric].items():
                train_value = train[metric].get(key, float("nan"))
                gap = (train_value - test_value if direction == "train_minus_test"
                       else test_value - train_value)
                row = dict(zip(index, key))
                row.update(test_split=test, metric=metric, train=train_value,
                           test=test_value, gap=gap,
                           rq3_included=int(row["k"] >= RQ3_MIN_K))
                rows.append(row)
    return pd.DataFrame(rows)


def run(exp_dir):
    exp_dir = Path(exp_dir)
    rows = []
    for done_path in sorted(exp_dir.glob("*/done.json")):
        meta = json.loads(done_path.read_text())
        eps = pd.read_csv(done_path.parent / "eval.csv")
        for (split, mode), group in eps.groupby(["split", "mode"]):
            row = dict(method=meta["method"], num_hosts=meta["num_hosts"],
                       k=meta["k"], seed=meta["seed"],
                       train_steps=meta.get("total_steps"),
                       split=split, mode=mode)
            row.update(summarise_episodes(group))
            rows.append(row)
    if not rows:
        print(f"no finished runs under {exp_dir}")
        return
    runs = pd.DataFrame(rows)
    runs.to_csv(exp_dir / "summary_runs.csv", index=False)

    grouped = runs.groupby(KEYS + ["split", "mode"])
    summary = grouped[METRICS].agg(["mean", "min", "max"])
    summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]
    summary.insert(0, "seeds", grouped["seed"].nunique())
    summary.reset_index().to_csv(exp_dir / "summary.csv", index=False)

    gaps = _gaps(runs)
    gaps.to_csv(exp_dir / "gaps_runs.csv", index=False)
    gap_group = gaps.groupby(
        KEYS + ["mode", "test_split", "metric", "rq3_included"])
    gap_summary = gap_group[["train", "test", "gap"]].agg(["mean", "min", "max"])
    gap_summary.columns = [f"{c}_{stat}" for c, stat in gap_summary.columns]
    gap_summary.insert(0, "seeds", gap_group["seed"].nunique())
    gap_summary.reset_index().to_csv(exp_dir / "gaps.csv", index=False)

    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.max_rows", 200,
                           "display.float_format", "{:.3f}".format):
        sampled = summary.xs("sampled", level="mode")
        print(sampled[["seeds"] + [f"{m}_mean" for m in METRICS]].T)
        sampled_gaps = gap_summary.xs("sampled", level="mode")
        print(sampled_gaps[["seeds", "train_mean", "test_mean", "gap_mean"]])
    print(f"wrote summary_runs.csv, summary.csv, gaps_runs.csv, gaps.csv in {exp_dir}")
