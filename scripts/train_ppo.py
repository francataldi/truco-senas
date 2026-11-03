import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.ppo.train import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ppo_3060.yaml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    cfg = TrainConfig.from_yaml(args.config)
    if args.out:
        cfg.out_dir = args.out
    if args.iterations:
        cfg.iterations = args.iterations
    if args.workers:
        cfg.workers = args.workers
    if args.fresh:
        cfg.resume = False
    print(train(cfg))


if __name__ == "__main__":
    main()
