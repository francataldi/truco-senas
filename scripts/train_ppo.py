import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.ppo.train import TrainConfig, train


def apply_overrides(cfg: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    if args.seed is not None:
        cfg.seed = args.seed
    if args.out:
        cfg.out_dir = args.out
    elif args.seed is not None and "{seed}" not in cfg.out_dir:
        cfg.out_dir = f"{cfg.out_dir}_seed{args.seed}"
    cfg.out_dir = cfg.out_dir.replace("{seed}", str(cfg.seed))
    if args.iterations:
        cfg.iterations = args.iterations
    if args.workers:
        cfg.workers = args.workers
    if args.fresh:
        cfg.resume = False
    return cfg


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/ppo_principal.yaml")
    p.add_argument("--out", default=None)
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--fresh", action="store_true")
    return p


def main() -> None:
    args = parser().parse_args()
    cfg = apply_overrides(TrainConfig.from_yaml(args.config), args)
    print(train(cfg))


if __name__ == "__main__":
    main()
