import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.arena import head_to_head
from agents.cfr import CFRAgent, save_table, train_parallel
from agents.heuristic import HeuristicAgent
from truco.rules import Rules


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=200_000)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--target", type=int, default=30)
    parser.add_argument("--out", default="runs/cfr/heads_up.pkl.gz")
    parser.add_argument("--eval-matches", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rules = Rules(num_players=2, target=args.target, signals=False)
    start = time.time()
    table = train_parallel(rules, args.iterations, args.workers, seed=args.seed)
    print(f"{len(table)} infosets en {time.time() - start:.0f}s")
    save_table(table, args.out, rules)
    agent = CFRAgent(table, seed=args.seed)
    stats = head_to_head(rules, [agent], [HeuristicAgent(seed=args.seed + 1)], args.eval_matches, seed=args.seed)
    print("CFR vs heuristico:", stats.to_dict())
    print(f"cobertura de infosets: {agent.hits / max(1, agent.hits + agent.misses):.3f}")


if __name__ == "__main__":
    main()
