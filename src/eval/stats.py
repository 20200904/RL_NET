"""Summarise the eval.csv files of an experiment directory.

Writes three tables next to the runs:
  summary_runs.csv  one row per (run, split, mode)
  summary.csv       mean / min / max over seeds
  gaps.csv          train-vs-test gaps per run and their mean over seeds

With few seeds only mean / min / max are reported. IQM, stratified bootstrap
CIs and probability of improvement (rliable) are added for the tier runs.
"""
import json
from pathlib import Path

import pandas as pd

KEYS = ["method", "num_hosts", "k"]
METRICS = [
    "solve_rate", "violation_rate", "violation_rate_among_solved",
    "safe_solve_rate", "mean_cost_c1", "mean_cost_c2", "mean_steps_solved",
    "scan_before_exploit", "c2_unscanned_frac",
]


def _ratio(num, den):
    return num / den if den else float("nan")


def summarise_episodes(eps):
    solved = eps[eps["solved"] == 1]
    return dict(
        episodes=len(eps),
        solve_rate=eps["solved"].mean(),
        violation_rate=(eps["cost"] > 0).mean(),
        violation_rate_among_solved=(
            (solved["cost"] > 0).mean() if len(solved) else float("nan")
        ),
        safe_solve_rate=((eps["solved"] == 1) & (eps["cost"] == 0)).mean(),
        mean_cost_c1=eps["cost_c1"].mean(),
        mean_cost_c2=eps["cost_c2"].mean(),
        mean_steps_solved=solved["steps"].mean() if len(solved) else float("nan"),
        scan_before_exploit=_ratio(eps["exploits_scanned"].sum(),
                                   eps["exploits"].sum()),
        c2_unscanned_frac=_ratio(eps["c2_unscanned"].sum(), eps["cost_c2"].sum()),
    )


def run(exp_dir):
    exp_dir = Path(exp_dir)
    rows = []
    for done_path in sorted(exp_dir.glob("*/done.json")):
        meta = json.loads(done_path.read_text())
        eps = pd.read_csv(done_path.parent / "eval.csv")
        for (split, mode), group in eps.groupby(["split", "mode"]):
            row = dict(method=meta["method"], num_hosts=meta["num_hosts"],
                       k=meta["k"], seed=meta["seed"], split=split, mode=mode)
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

    # gaps: positive always means "worse on the unseen networks"
    wide = runs.pivot_table(index=KEYS + ["seed", "mode"], columns="split",
                            values=["solve_rate", "violation_rate"])
    gap_rows = []
    for test in ("test_a", "test_b"):
        if ("solve_rate", test) not in wide.columns:
            continue
        gaps = pd.DataFrame(dict(
            test_split=test,
            train_solve_rate=wide[("solve_rate", "train")],
            test_solve_rate=wide[("solve_rate", test)],
            capability_gap=wide[("solve_rate", "train")] - wide[("solve_rate", test)],
            train_violation_rate=wide[("violation_rate", "train")],
            test_violation_rate=wide[("violation_rate", test)],
            safety_gap=(wide[("violation_rate", test)]
                        - wide[("violation_rate", "train")]),
        )).reset_index()
        gap_rows.append(gaps)
    gaps = pd.concat(gap_rows, ignore_index=True)
    gaps.to_csv(exp_dir / "gaps_runs.csv", index=False)
    gap_cols = [c for c in gaps.columns if c not in KEYS + ["seed", "mode", "test_split"]]
    gap_group = gaps.groupby(KEYS + ["mode", "test_split"])
    gap_summary = gap_group[gap_cols].mean()
    gap_summary.insert(0, "seeds", gap_group["seed"].nunique())
    gap_summary.reset_index().to_csv(exp_dir / "gaps.csv", index=False)

    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.float_format", "{:.3f}".format):
        show = ["seeds", "solve_rate_mean", "violation_rate_mean",
                "violation_rate_among_solved_mean", "safe_solve_rate_mean",
                "mean_cost_c1_mean", "mean_cost_c2_mean", "mean_steps_solved_mean"]
        print(summary[show])
        print(gap_summary)
    print(f"wrote summary_runs.csv, summary.csv, gaps_runs.csv, gaps.csv in {exp_dir}")
