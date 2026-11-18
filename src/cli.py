"""Entry point: python -m src.cli <command>

Commands are added phase by phase; only implemented ones are listed.
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="python -m src.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- Phase 0 ----
    p = sub.add_parser("check-env", help="load NASim, random policy, steps/sec")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--out", type=Path, default=Path("results/phase0"))

    p = sub.add_parser("sb3-check", help="Phase 0 SB3 PPO reference run")
    p.add_argument("--scenario", default="tiny",
                   help="benchmark name (tiny, small) or genN (e.g. gen6)")
    p.add_argument("--fully-obs", action="store_true")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--net-seed", type=int, default=1)
    p.add_argument("--step-limit", type=int, default=400)
    p.add_argument("--out", type=Path, default=Path("results/phase0"))

    sub.add_parser("diag-det-eval",
                   help="Phase 0 diagnosis: argmax vs sampled eval on tiny")

    # ---- Phase 1 ----
    p = sub.add_parser("gen-scenarios", help="generate networks and splits")
    p.add_argument("--hosts", type=int, nargs="+", default=[6, 8])
    p.add_argument("--root", type=Path, default=Path("scenarios"))

    p = sub.add_parser("calibrate", help="random solve rate vs step limit")
    p.add_argument("--hosts", type=int, nargs="+", default=[6, 8])
    p.add_argument("--out", type=Path, default=Path("results/phase1"))
    p.add_argument("--paths-only", action="store_true",
                   help="only compute shortest safe / unconstrained paths")

    p = sub.add_parser("validate-ppo", help="hand-written PPO vs SB3 on tiny")
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out", type=Path, default=Path("results/phase1/validate_ppo"))

    p = sub.add_parser("run-grid", help="run every job of a config (resumable)")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1)

    p = sub.add_parser("re-eval",
                       help="rewrite eval.csv of finished runs from saved models")
    p.add_argument("--config", type=Path, required=True)

    p = sub.add_parser("convergence", help="learning curves, steps to converge")
    p.add_argument("--exp", type=Path, required=True)

    sub.add_parser("basics", help="rule agents, observation and cost checks")
    sub.add_parser("reward-audit", help="reward redesign stage 1 (no training)")
    p = sub.add_parser("reward-grid", help="reward redesign stage 2 (no training)")
    p.add_argument("--rescore", action="store_true",
                   help="only rescore saved rollouts and path searches")
    sub.add_parser("reward-report",
                   help="regenerate the blocks in report/reward_redesign.md")

    p = sub.add_parser("stats", help="summarise eval.csv files of an experiment")
    p.add_argument("--exp", type=Path, required=True)

    p = sub.add_parser("readme-tables",
                       help="regenerate the generated blocks in README.md")

    args = parser.parse_args()
    if args.command == "check-env":
        from src.phase0 import check_env
        check_env.run(num_steps=args.steps, out_dir=args.out)
    elif args.command == "sb3-check":
        from src.phase0 import sb3_check
        sb3_check.run(args.scenario, args.fully_obs, args.seeds, args.steps,
                      out_dir=args.out, net_seed=args.net_seed,
                      step_limit=args.step_limit)
    elif args.command == "diag-det-eval":
        from src.phase0 import diag_det_eval
        diag_det_eval.run()
    elif args.command == "gen-scenarios":
        from src.envs import generate
        for num_hosts in args.hosts:
            specs, tried = generate.build_splits(num_hosts)
            path = generate.save_splits(specs, tried, num_hosts, args.root)
            print(f"wrote {path}: {len(specs)} networks, seeds tried {tried}")
    elif args.command == "calibrate":
        from src.eval import calibrate
        for num_hosts in args.hosts:
            if args.paths_only:
                calibrate.path_lengths(num_hosts, out_dir=args.out)
            else:
                calibrate.run(num_hosts, out_dir=args.out)
                calibrate.path_lengths(num_hosts, out_dir=args.out)
    elif args.command == "validate-ppo":
        from src.eval import validate_ppo
        validate_ppo.run(args.steps, args.seeds, args.out)
    elif args.command == "run-grid":
        from src import runner
        cfg = runner.load_config(args.config)
        runner.run_grid(runner.expand_jobs(cfg), args.workers)
    elif args.command == "re-eval":
        from src import runner
        for job in runner.expand_jobs(runner.load_config(args.config)):
            print(runner.reeval_job(job), flush=True)
    elif args.command == "convergence":
        from src.eval import convergence
        convergence.run(args.exp)
    elif args.command == "reward-audit":
        from src.eval import reward_audit
        reward_audit.run()
    elif args.command == "reward-grid":
        from src.eval import reward_grid
        reward_grid.rescore() if args.rescore else reward_grid.run()
    elif args.command == "reward-report":
        from src.eval import reward_report
        reward_report.run()
    elif args.command == "basics":
        from src.eval import basics
        basics.run()
    elif args.command == "stats":
        from src.eval import stats
        stats.run(args.exp)
    elif args.command == "readme-tables":
        from src.eval import readme_tables
        readme_tables.run()


if __name__ == "__main__":
    main()
