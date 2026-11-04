import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from agents.crossplay import run_crossplay
from agents.plots import plot_intervals, plot_matrix
from agents.registry import spec_label, spec_rules

CELL_FIELDS = ("a", "b", "tipo", "partidos", "victorias", "tasa", "ci95_lo", "ci95_hi", "dif_puntos_por_mano")


def resolve(args) -> tuple[list[str], list[str]]:
    specs = list(args.specs)
    for run in args.runs:
        path = Path(run) / args.checkpoint
        if not path.exists():
            raise SystemExit(f"no existe {path}")
        specs.append(f"ppo:{path.as_posix()}")
    names = []
    for spec in specs:
        label = spec_label(spec)
        names.append(label.split("@")[0] if args.short_names else label)
    if len(set(names)) != len(names):
        names = [spec_label(s) for s in specs]
    return names, specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=[])
    parser.add_argument("--specs", nargs="*", default=[])
    parser.add_argument("--checkpoint", default="last.pt")
    parser.add_argument("--opponent", default="heuristic")
    parser.add_argument("--matches", type=int, default=200)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--marginal-hands", type=int, default=1000)
    parser.add_argument("--signal-hands", type=int, default=3000)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--short-names", action="store_true")
    parser.add_argument("--out", default="runs/crossplay")
    args = parser.parse_args()
    names, specs = resolve(args)
    if len(specs) < 2:
        raise SystemExit("el cross-play necesita al menos dos agentes")
    rules = spec_rules(specs[0])
    for spec in specs[1:]:
        if spec_rules(spec) != rules:
            raise SystemExit(f"{spec} se entrenó con otras reglas que {specs[0]}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = run_crossplay(
        rules,
        names,
        specs,
        opponent=args.opponent,
        matches=args.matches,
        seed=args.seed,
        workers=args.workers,
        marginal_hands=args.marginal_hands,
        signal_hands=args.signal_hands,
        bootstrap=args.bootstrap,
    )
    (out / "crossplay.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out / "crossplay.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(CELL_FIELDS))
        writer.writeheader()
        writer.writerows(summary["celdas"])
    matrix = np.array([[np.nan if v is None else v for v in row] for row in summary["matriz"]])
    plot_matrix(names, names, matrix, out / "crossplay_matriz.png", f"Pareja (fila, columna) contra {args.opponent}", colorbar="victorias de la pareja")
    labels, values, lows, highs = [], [], [], []
    for key, label in (("self_play", "self-play"), ("cross_play", "cross-play"), ("self_play_mezclado", "self-play con señas mezcladas")):
        if key in summary["agregado"]:
            entry = summary["agregado"][key]
            labels.append(label)
            values.append(entry["p"])
            lows.append(entry["ci95"][0])
            highs.append(entry["ci95"][1])
    plot_intervals(
        labels,
        values,
        lows,
        highs,
        out / "crossplay.png",
        f"Cross-play contra self-play (rival: {args.opponent})",
        "tasa de victoria con IC 95% de Wilson",
        reference=0.5,
    )
    print(json.dumps({k: summary[k] for k in summary if k not in ("celdas", "matriz", "diccionarios")}, indent=2, ensure_ascii=False))
    print(f"resultados en {out}")


if __name__ == "__main__":
    main()
