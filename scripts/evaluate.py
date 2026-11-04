import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.arena import head_to_head
from agents.base import limit_threads
from agents.registry import make_agent
from truco.rules import Rules


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", default="heuristic")
    parser.add_argument("--matches", type=int, default=500)
    parser.add_argument("--players", type=int, default=4, choices=(2, 4))
    parser.add_argument("--target", type=int, default=30)
    parser.add_argument("--spy", type=float, default=0.2)
    parser.add_argument("--no-signals", action="store_true")
    parser.add_argument("--scramble-a", action="store_true")
    parser.add_argument("--scramble-b", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    rules = Rules(
        num_players=args.players,
        target=args.target,
        signals=not args.no_signals and args.players == 4,
        spy_prob=args.spy,
    )
    a = make_agent(args.a, seed=args.seed)
    b = make_agent(args.b, seed=args.seed + 1)
    limit_threads()
    stats = head_to_head(rules, [a], [b], args.matches, seed=args.seed, scramble_a=args.scramble_a, scramble_b=args.scramble_b)
    result = {"a": args.a, "b": args.b, "rules": rules.to_dict(), **stats.to_dict()}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
