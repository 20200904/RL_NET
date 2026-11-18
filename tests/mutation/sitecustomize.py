# Mutation check for test_train_and_eval_identical_across_processes.
#
# Restores NASim's original (hash-order dependent) firewall generator in every
# Python process that has this directory on PYTHONPATH and MUTANT set:
#   MUTANT=check    original firewall, stored-digest check left on
#   MUTANT=nocheck  original firewall, stored-digest check disabled, so only
#                   the cross-process comparison can catch it
# The test must FAIL in both modes. From the repo root (Git Bash):
#   MUTANT=nocheck PYTHONPATH="$(cygpath -w $PWD);$(cygpath -w $PWD/tests/mutation)" \
#     .venv/Scripts/python -m pytest \
#     tests/test_phase1.py::test_train_and_eval_identical_across_processes
import os
if os.environ.get("MUTANT"):
    from nasim.scenarios.generator import ScenarioGenerator
    from src.envs import generate as g
    g.FixedTableGenerator._generate_firewall = ScenarioGenerator._generate_firewall
    if os.environ["MUTANT"] == "nocheck":
        g.verified_scenario = lambda spec: g.make_scenario(spec.seed, spec.num_hosts)
        import src.envs.multi_env as me
        me.verified_scenario = g.verified_scenario
