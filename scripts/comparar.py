import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.behavior import SOURCES, compare
from truco.rules import Rules

NOMBRES = {
    "envido": "envido",
    "cartas": "ganar jugando las cartas",
    "truco_rechazado": "el rival no quiso el truco",
    "mazo": "el rival se fue al mazo",
}


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", default="heuristic")
    parser.add_argument("--matches", type=int, default=1000)
    parser.add_argument("--players", type=int, default=4, choices=(2, 4))
    parser.add_argument("--target", type=int, default=30)
    parser.add_argument("--spy", type=float, default=0.2)
    parser.add_argument("--no-signals", action="store_true")
    parser.add_argument("--simple", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ruido", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    rules = Rules(num_players=args.players, target=args.target, signals=not args.no_signals and args.players == 4, spy_prob=args.spy)
    noise = json.loads(args.ruido) if args.ruido else None
    result = compare(rules, args.a, args.b, args.matches, seed=args.seed, duplicate=not args.simple, workers=args.workers, noise=noise)
    a, b = result["equipo_a"], result["equipo_b"]
    print(f"{args.a} contra {args.b}: gana {_pct(result['gana_a'])} de {result['partidos']} partidos")
    print(f"diferencia por mano: {result['diferencia_por_mano']['diff']:+.3f} {result['diferencia_por_mano']['ci95']}")
    for s in SOURCES:
        d = result["diferencia_por_fuente"][s]
        print(f"  {NOMBRES[s]:28s} {d['diff']:+.3f} {d['ci95']}  ({a['puntos_por_mano_por_fuente'][s]:.3f} contra {b['puntos_por_mano_por_fuente'][s]:.3f})")
    for k in ("envido", "truco"):
        for m in a[k]:
            va, vb = a[k][m], b[k][m]
            fmt = (lambda x: "-" if x is None else f"{x:.1f}") if m == "envido_medio_al_cantar" else _pct
            print(f"  {k} {m:28s} {fmt(va):>8s} {fmt(vb):>8s}")
    print(f"  se va al mazo {_pct(a['se_va_al_mazo']):>27s} {_pct(b['se_va_al_mazo']):>8s}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
