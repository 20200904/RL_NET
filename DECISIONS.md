# Decisions

Design choices and changes, with date and reason. Newest last.

## 2026-11-16: decisions approved by the user after gate 0

1. **Exploit / privilege-escalation table fixed across all networks.**
   NASim's generator re-draws it per seed, so an action index would mean a
   different exploit in each network and the observation does not describe
   exploits. Implemented as a subclass of NASim's generator
   (`FixedTableGenerator`); NASim is not modified.
2. **Observation memory wrapper added** (`ObsMemory`). Partially observable
   NASim returns only the last action's result, so a memoryless MLP could
   never act on a scan. Stated in the README method section.
3. **Topology stays fixed** per host count and is listed as a limitation.
   NASim's generator derives subnets and topology from the host count alone.
4. **Host count decided in Phase 1** from M0 / M1 measurements at 6 and 8 hosts.
5. **Gate 1 criteria changed** to: random solve rate <= 20% within the step
   limit; M1 violation rate between 20% and 80%; a zero-cost path in every network.
6. **Tier 1 enlarged** to 10 seeds and K in {4, 16, 32}.

## 2026-11-16: design details fixed while implementing Phase 1

All of these were fixed before any M1 run.

- **Fragility is a global rule, not a per-network one**: a host is fragile if
  it runs `os_1` together with `srv_1`. The plan said each network gets a
  fragile rule by seed, but an unobserved rule that changes per network could
  not be inferred from scans. What the seed changes is which hosts match.
- **Risky exploits** are the two that grant root in one step (`e_srv_1`,
  `e_srv_3_os_1`). They are the shortcut a reward-only agent prefers.
- **Fragile fraction is controlled by rejection**: train and test A accept
  networks with 15-40% fragile hosts, test B 45-75%.
- **Costs are charged for the attempt**, independent of NASim's result,
  including connection and permission errors.
- **The out-of-scope host** is never a goal and never the DMZ entry host.
- **Zero-cost path check**: breadth-first search over NASim states; it also
  gives the length of the shortest zero-cost path.
- **Networks are not stored as yaml.** `scenarios/splits_h*.json` stores the
  seed and the derived facts of each network; the network is regenerated from
  the seed (deterministic). Simpler than writing NASim's yaml format.
- **Step limit lives in the wrapper**, not in the scenario.
- **Resuming is per run.** A run that died is rerun from scratch; finished
  runs (`done.json`) are skipped. Runs take minutes, and this keeps every
  finished run reproducible from its seed. No mid-run checkpoints.
- **Truncated episodes are treated as terminal in GAE** (no bootstrap), as in
  the reference CleanRL implementation.
- **Firewall restrictiveness 2**, `random_goal=True`, 4 services, 2 OS,
  2 processes, all exploits deterministic.

## 2026-11-16: step limit 100

Numbers below are from the regenerated calibration (after the
reproducibility fix); see the re-check at the end of this file.
From `results/phase1/calibration_h*.csv`: at 100 steps the random policy
solves 8.2% (6 hosts) and 1.9% (8 hosts) of episodes on the training pool,
below the 20% criterion. At 150 steps the 6-host value is already above 20%
(22.9%). The longest shortest zero-cost path in the training pools is 7
(6 hosts) and 8 (8 hosts) actions, so 100 steps leaves more than ten times that.

## 2026-11-16: gate 1 decisions by the user

- **8 hosts, step limit 100: confirmed.**
- **Approved as implemented:** the global fragile rule, regenerating networks
  from their seed instead of storing yaml, and no mid-run checkpoints (all
  three are described in the section above).
- **Violation-rate criterion moved to K=16 and K=32, judged after
  convergence.** Only the lower bound (20%) is required. The upper bound
  (80%) is informational: if M1 exceeds it, this is reported and the fragile
  fraction is *not* lowered in response.
- **Convergence probe before Phase 2:** M1 on 8 hosts, K=16 and K=32, 1M and
  2M steps, 3 seeds, to fix the training steps per K.

## 2026-11-16: post-hoc decisions, made after looking at Phase 1 data

HYPOTHESES.md is not edited. The following were decided after seeing the
Phase 1 results and are therefore not pre-registered.

- **K=4 is excluded from the RQ3 gap comparison.** Reason: at K=4, M1 solved
  100% of its 4 training networks and about as many unseen networks as the
  random policy (`results/phase1/runs/summary.csv`). It memorises, so its
  "gaps" compare a goal-reaching policy with a failing one and say nothing
  about whether safety or capability transfers. K=4 stays in tier 1 as a
  memorisation baseline. H3 is evaluated on K=16 and K=32 only. The
  secondary expectation in HYPOTHESES.md ("both gaps shrink as K grows from
  4 to 32") can no longer be read as written; it is reported for K=16 vs
  K=32, with the K=4 numbers shown separately.
- **Metrics added** to the evaluation code and README tables. Reason: the
  violation rate saturates near 100% for policies that flail, so it cannot
  separate "one cost event" from "thirty". Added:
  - cost events per episode, C1 and C2 separately
  - cost events per attack attempt (exploit or privilege escalation; these
    are the only actions that can incur a cost, and C1 can come from a
    privilege escalation, so both are in the denominator)
  - violation rate among goal-reaching episodes only
  - generalization gaps for each of these three and for the solve rate
  The pre-registered headline metric (safe solve rate) and the definitions
  in HYPOTHESES.md are unchanged.
- **The 300k-step K=16/32 probe** (1 seed) was an unplanned exploratory run.
  It was made before the reproducibility fix, was deleted, and is replaced
  by the convergence probe (`results/phase1/converge/`). It remains in git
  history at `413b939`.

## 2026-11-16: bug found and fixed, Phase 1 results regenerated

**Bug.** The same network seed produced a different firewall in every Python
process. NASim's generator draws permitted services with
`np.random.choice(list(a_set_of_strings))`, and the iteration order of a set
of strings depends on Python's per-process string hash seed. Found when a
re-evaluation of saved models did not reproduce the original `eval.csv`, not
even for the random policy.

**What it affected.** Every Phase 1 run made before the fix (commit
`413b939`): the K=4 grid, the 300k-step K=16/32 probe, calibration and path
lengths. Within one process the networks were consistent, so each run
trained and evaluated on *some* valid networks from the intended
distribution. But the networks differed between runs and between seeds, the
recorded zero-cost path had been checked on a different firewall than the
one used, and "same test set for every run" was not true. Those numbers were
reported at gate 1 and are superseded.

**Fix.** `FixedTableGenerator._generate_firewall` reimplements NASim's rules
with sorted candidates (NASim itself is still unmodified). Each network's
digest is stored in `scenarios/splits_h*.json` and checked every time the
network is rebuilt, so a mismatch raises instead of passing silently. A
regression test builds the networks under three different hash seeds.
Evaluation also pins torch to one thread.

**Regenerated.** Splits, calibration, path lengths and the K=4 grid were
deleted and rerun; the 300k probe was dropped (the convergence probe replaces
it). The earlier versions remain in git history at `413b939`. The step-limit
decision was rechecked on the new calibration and stands.

## 2026-11-16: host count 8 (confirmed by the user at gate 1)

The numbers that supported this at gate 1 came from runs made before the
reproducibility fix and are discarded. The decision is re-checked on the
regenerated runs at the end of this file.

## 2026-11-17: re-check of the gate 1 decisions on regenerated numbers

Every number at gate 1 came from runs made before the reproducibility fix.
Each decision is re-checked here. Sources: `results/phase1/calibration_h*.csv`,
`results/phase1/path_lengths_h*.csv`, `results/phase1/runs/summary_runs.csv`
(K=4, 300k steps, 3 seeds) and `results/phase1/converge/2m/summary.csv`
(8 hosts, K=16/32, 2M steps, 3 seeds). All rates are sampled evaluation.

| decision | status | numbers |
|---|---|---|
| step limit 100 | **kept** | random solve rate at 100 steps: 8.2% (6 hosts), 1.9% (8 hosts), both <= 20%; at 150 steps 6 hosts is 22.9% |
| 8 hosts | **kept** | random solve 1.9% vs 8.2%. K=4 M1 training-pool violation rate: 6 hosts 0.8% (0.0–2.5%) fails the 20% floor, 8 hosts 35.8% (10.0–52.5%) passes. With the criterion moved to converged K=16/32: 8 hosts 79.0% (K=16) and 65.2% (K=32), above the floor. 6 hosts was not run at K=16/32. Shortcut-network share in the training pool is similar (25.0% vs 21.9%), so the K=4 difference still looks like a sample effect |
| K=4 policies memorise | **kept** | 8 hosts: training-pool solve 100%, test A 1.8%, random policy on test A 2.2%; 6 hosts: test A 4.9%, random 6.0% |
| K=4 excluded from RQ3 | **kept for K=4, but the reason now also applies to K=16 and K=32** | converged M1 (2M steps): test A solve rate 5.9% at K=16 and 8.8% at K=32, against 99.8% and 99.9% on the training pool, and 2.2% for the random policy. Not decided; raised with the user |
| violation floor (>= 20%) at converged K=16/32 | **passes** | 79.0% (74.4–87.5%) at K=16, 65.2% (62.8–68.8%) at K=32. The informational 80% ceiling is exceeded by one K=16 seed (87.5%) |

The open problem recorded at gate 1 ("K=16/32 does not converge in 300k
steps") came from the discarded single-seed probe. With 3 seeds and up to 2M
steps it converges; steps per K are in `docs/compute_budget.md`.

## 2026-11-17: reward redesign plan adopted by the user

After the basic checks (`docs/notes_basics.md`) showed that the task is
solvable from observations and that, under the current reward, a policy
memorised for the training networks earns far more than a general strategy,
the user adopted `docs/plans/REWARD_REDESIGN_PLAN.md`. Fixed by the plan:
C1 and C2 stay out of the reward; on reward alone the unsafe path must still
pay more; HYPOTHESES.md is not edited; existing results are kept as the
`reward=uniform` condition. The reward is not changed before gate 2.

## 2026-11-18: reward redesign stage 2, decisions by the user at gate 1

The stage 2 grid was set by the user after the stage 1 audit, before any
candidate was scored.

1. **Subnet scans keep their own action and cost the same as the other
   scans** (OS, service, process): one reconnaissance cost axis. Reason:
   scanning, host discovery included, is a basic and essential part of any
   engagement, so reconnaissance should never be a loss. Known side effect:
   with a low scan cost, a subnet scan that finds hosts can be net positive
   (discovery reward above the scan cost).
2. **The general-strategy baseline is the rule agent "scan, then attack,
   avoiding out-of-scope and fragile targets"** (`scan_safe`, robot B).
   Reason: the goal is a policy that is both general and safe. Robot A
   (`scan`) is still scored and kept in `results/reward/grid.csv`;
   candidates are ranked by robot B.
3. **Scoring uses the discounted return with gamma = 0.99.** Reason: it is
   what PPO maximises, so it predicts what PPO will prefer. Because costs are
   unequal and returns are discounted, the shortest path is not necessarily
   the highest-return path, so the optimal and optimal-safe paths are
   recomputed for every candidate (see `src/eval/reward_grid.py`).
4. **Target reward axis {10, 100}.** Reason: scoring is cheap, and 100
   matches NASim's own benchmark scale. Δ is reported in points and as a
   fraction of the target reward.
5. **Failed-attack penalty axis {0, 2}**, charged on top of the action cost.
   A failed attack is an exploit or privilege escalation after which the
   agent's access level on the target host is not higher than before (no new
   access); this includes connection and permission errors and attacks on
   hosts already held at that level. Reason: failed attacks carry real
   operational risk (noise, crashes). This penalty cannot change Δ: a
   full-information optimal path never contains a failed attack. It only
   changes the scores of the rule agents, the random policy and the M1
   trajectories.

Grid: 4 scan costs x 3 attack costs x 2 discovery x 2 non-target root x
2 target x 2 failed-attack penalty = 192 candidates.
