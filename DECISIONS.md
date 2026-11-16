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

From `results/phase1/calibration_h*.csv`: at 100 steps the random policy
solves 8.1% (6 hosts) and 1.7% (8 hosts) of episodes on the training pool,
below the 20% criterion. At 150 steps the 6-host value is already above 20%.
The longest shortest zero-cost path in the training pools is 7 (6 hosts) and
8 (8 hosts) actions, so 100 steps leaves more than ten times that.

## 2026-11-16: host count 8 (proposed, pending user confirmation at gate 1)

Measured with M1, K=4, 3 seeds, sampled evaluation on the training networks
(`results/phase1/runs/summary.csv`):

| criterion | 6 hosts | 8 hosts |
|---|---|---|
| random solve rate <= 20% | pass | pass |
| M1 violation rate in 20-80% | fail (below 20%) | pass |
| zero-cost path in every network | pass (by construction) | pass (by construction) |

Caveat: with K=4 the violation rate is decided by which four networks are
used. Structurally the two pools are alike: the share of training networks
where ignoring costs gives a shorter path is about the same for both
(`results/phase1/path_lengths_h*.csv`). So the criterion separates the two
host counts mostly by chance; 8 hosts is chosen because it passes as
measured and its random-policy solve rate is lower.

## Open problem found in Phase 1 (not decided)

At K=16 and K=32 an exploratory single-seed run with the same 300k steps
(`results/phase1/probe_k/summary.csv`) gives M1 violation rates above the 80%
ceiling and episodes far longer than the shortest path, i.e. the policy has
not converged. The second gate criterion is therefore met at K=4 but not at
larger K with the current step budget. Options are in the gate 1 report.
