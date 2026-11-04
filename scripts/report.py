import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.report import build_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", default="runs/reporte")
    parser.add_argument("--signal-hands", type=int, default=4000)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = build_report(args.runs, args.out, signal_hands=args.signal_hands, bootstrap=args.bootstrap, seed=args.seed)
    print(f"{len(report['runs'])} corridas, {len(report['figuras'])} figuras en {args.out}")
    for figure in report["figuras"]:
        print(f"  {figure['archivo']}: {figure['titulo']}")


if __name__ == "__main__":
    main()
