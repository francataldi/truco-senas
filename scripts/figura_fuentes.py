import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.behavior import plot_sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", nargs="+", required=True)
    parser.add_argument("--nombres", nargs="+", required=True)
    parser.add_argument("--titulo", default=None)
    parser.add_argument("--eje", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if len(args.json) != len(args.nombres):
        parser.error("hace falta un nombre por cada json")
    results = [(name, json.loads(Path(p).read_text(encoding="utf-8"))) for name, p in zip(args.nombres, args.json)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plot_sources(results, args.out, title=args.titulo, xlabel=args.eje)


if __name__ == "__main__":
    main()
