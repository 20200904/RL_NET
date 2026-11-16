# Hypotheses

Committed before the first Phase 1 experiment. Not edited afterwards except
for typos; any change of design goes into `DECISIONS.md` with a date and a
reason. Hypotheses that turn out wrong are reported as wrong.

## What had been run before this commit

Only Phase 0 measurements, none of which involve the cost signal or the
generated train/test networks:

- random-policy throughput (`results/phase0/check_env.csv`)
- SB3 PPO reference runs on `tiny` and on one generated 6-host network
  (`results/phase0/sb3_summary.csv`, `results/phase0/long/sb3_summary.csv`)
- an argmax-vs-sampled evaluation diagnosis on `tiny`
  (`results/phase0/diag_det_eval.csv`)

## Definitions

- **Episode cost**: number of C1 events plus number of C2 events in an episode.
  - C1 (out of scope): an exploit or privilege escalation aimed at the
    out-of-scope host.
  - C2 (service disruption): a risky exploit aimed at a fragile host,
    whether or not it succeeds.
- **Solved**: all sensitive hosts compromised with root access within the step limit.
- **Violation**: an episode with cost > 0. **Violation rate**: fraction of
  episodes that are violations.
- **Cost budget** `d = 0.1` expected episode cost (the PPO-Lagrangian target).
- **Safe solve** (headline metric): solved AND episode cost <= d. Costs are
  integers, so this means solved with zero cost.
- **Generalization gap**: value on the training pool minus value on a held-out
  test set (solve rate), or test minus train (violation rate), so that a
  positive gap always means "worse on unseen networks".

## Headline evaluation mode: sampled actions

Both argmax and sampled-action evaluation are recorded. The headline is the
**sampled (stochastic) policy**.

Reason, from `results/phase0/diag_det_eval.csv`: on `tiny`, argmax evaluation
scored a 0% solve rate for three of four trained SB3 PPO policies while the
same policies solved 100% of episodes when sampled. In every failing case the
argmax policy used exactly one distinct action for all 1000 steps. A failed or
no-effect action leaves the state unchanged, the observation is then
unchanged, and argmax picks the same action forever. Argmax therefore turns
one bad state into a total failure and measures something other than the
policy PPO optimises. Sampled evaluation needs several episodes per network;
the number is fixed in the evaluation config before the tier runs.

## Environment parameters still open at this commit

Host count (6 or 8), step limit and the fragile-host fraction are calibrated
in Phase 1 using only the random policy (M0) and reward-only PPO (M1), against
these criteria agreed with the user beforehand:

- random-policy solve rate within the step limit <= 20%
- M1 violation rate between 20% and 80%
- a zero-cost path to the goal exists in every network

The chosen values and the reason are recorded in `DECISIONS.md` before any
constrained method (M2-M4) is run.

## Research questions and hypotheses

**RQ1.** In what fraction of goal-reaching episodes does reward-only PPO (M1)
incur a safety cost?

- H1: a substantial fraction. Supported if, among M1's solved episodes on the
  training pool, at least 20% have cost > 0.

**RQ2.** How much solve rate do fixed penalties (M2a, M2b) and PPO-Lagrangian
(M3) give up, and how much cost do they remove?

- H2: at a comparable cost level, M3 solves more. Supported if M3's safe solve
  rate (IQM over seeds) is higher than that of the better fixed-penalty
  variant and the probability of improvement of M3 over it has a 95% CI
  above 0.5. Refuted if the CI lies below 0.5; otherwise unclear.

**RQ3.** On unseen networks, which drops more: capability or safety? Does
training on more networks (K) shrink the gaps?

- H3: the safety gap (violation rate, test minus train) is larger than the
  capability gap (solve rate, train minus test) on test set A. Supported if the
  95% bootstrap CI of (safety gap - capability gap) is above 0.
- Secondary expectation: both gaps shrink as K grows from 4 to 32.

**RQ4.** In violating episodes, what did the agent skip?

- H4: most C2 events happen without a prior scan. Supported if, in more than
  50% of C2 events, the agent had not yet observed both the OS and the
  services of the target host.

## Reporting rules

- Every number in the README and the report is generated from csv files under
  `results/` by a script.
- Cells for experiments that were not run say "not run".
- Results from 5 or fewer seeds are labelled preliminary.
