"""Entry point: python -m src.cli <command>

Commands are added phase by phase; only implemented ones are listed.
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="python -m src.cli")
    sub = parser.add_subparsers(dest="command", required=True)

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

    args = parser.parse_args()
    if args.command == "check-env":
        from src.phase0 import check_env
        check_env.run(num_steps=args.steps, out_dir=args.out)
    elif args.command == "sb3-check":
        from src.phase0 import sb3_check
        sb3_check.run(args.scenario, args.fully_obs, args.seeds, args.steps,
                      out_dir=args.out, net_seed=args.net_seed,
                      step_limit=args.step_limit)


if __name__ == "__main__":
    main()
