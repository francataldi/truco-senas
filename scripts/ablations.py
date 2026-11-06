import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.ablations import CONDITIONS, run_ablations
from agents.plots import plot_intervals
from agents.registry import spec_rules
from truco.rules import Rules

LABELS = {
    "normal": "normal",
    "compañero_mezclado": "señas del compañero mezcladas",
    "rivales_mezclados": "señas espiadas mezcladas",
    "ambos_mezclados": "compañero y rivales mezclados",
    "rivales_ocultos": "sin espiar a los rivales",
    "seña_constante": "seña constante",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--opponent", default=None)
    parser.add_argument("--matches", type=int, default=400)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--spy", type=float, default=None)
    parser.add_argument("--marginal-hands", type=int, default=2000)
    parser.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duplicado", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    spec_a = args.checkpoint if ":" in args.checkpoint else f"ppo:{Path(args.checkpoint).as_posix()}"
    spec_b = args.opponent or spec_a
    trained = spec_rules(spec_a)
    if trained is None or not trained.signals:
        raise SystemExit("las ablaciones necesitan un agente entrenado con señas")
    rules = trained if args.spy is None else Rules(**{**trained.to_dict(), "spy_prob": args.spy})
    path = Path(spec_a.partition(":")[2])
    out = Path(args.out) if args.out else path.parent / "ablaciones"
    out.mkdir(parents=True, exist_ok=True)
    summary = run_ablations(
        rules,
        spec_a,
        spec_b,
        args.matches,
        seed=args.seed,
        workers=args.workers,
        conditions=tuple(args.conditions),
        marginal_hands=args.marginal_hands,
        duplicate=args.duplicado,
    )
    summary["run"] = path.parent.name
    (out / "ablaciones.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = []
    for condition, entry in summary["condiciones"].items():
        rows.append(
            {
                "condicion": condition,
                "partidos": entry["matches"],
                "tasa": entry["win_rate"],
                "ci95_lo": entry["ci95"][0],
                "ci95_hi": entry["ci95"][1],
                "dif_puntos_por_mano": entry["point_diff_per_hand"],
            }
        )
    with open(out / "ablaciones.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    conditions = list(summary["condiciones"])
    plot_intervals(
        [LABELS.get(c, c) for c in conditions],
        [summary["condiciones"][c]["win_rate"] for c in conditions],
        [summary["condiciones"][c]["ci95"][0] for c in conditions],
        [summary["condiciones"][c]["ci95"][1] for c in conditions],
        out / "ablaciones.png",
        f"Ablaciones de señas · {summary['run']} contra {'sí mismo' if spec_b == spec_a else spec_b}",
        "tasa de victoria con IC 95% de Wilson",
        reference=0.5 if summary["referencia_valores"] == "simetria_0.5" else summary["condiciones"].get("normal", {}).get("win_rate"),
        highlight=[c == "normal" for c in conditions],
    )
    brief = {"condiciones": {c: (e["win_rate"], e["ci95"]) for c, e in summary["condiciones"].items()}, "valores": summary["valores"]}
    print(json.dumps(brief, indent=2, ensure_ascii=False))
    print(f"resultados en {out}")


if __name__ == "__main__":
    main()
