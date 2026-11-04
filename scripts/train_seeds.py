import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.ppo.train import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ppo_semillas.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--out", default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    for seed in args.seeds:
        cfg = TrainConfig.from_yaml(args.config)
        cfg.seed = seed
        template = args.out or cfg.out_dir
        if "{seed}" not in template:
            template = f"{template}/seed_{{seed}}"
        cfg.out_dir = template.replace("{seed}", str(seed))
        if args.iterations:
            cfg.iterations = args.iterations
        if args.workers:
            cfg.workers = args.workers
        if args.fresh:
            cfg.resume = False
        print(f"=== semilla {seed} -> {cfg.out_dir}", flush=True)
        print(train(cfg), flush=True)


if __name__ == "__main__":
    main()
