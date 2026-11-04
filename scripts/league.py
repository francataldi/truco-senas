import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.league import Entrant, compatible, parse_entrant, round_robin, run_checkpoints, summarize_league
from agents.plots import plot_intervals, plot_matrix
from agents.registry import spec_label
from truco.rules import Rules

RATING_FIELDS = ("nombre", "spec", "rating", "elo", "elo_lo", "elo_hi", "p_mejor", "partidos", "victorias", "tasa")
PAIR_FIELDS = ("a", "b", "partidos", "victorias_a", "tasa_a", "ci95_lo", "ci95_hi", "esperado_bt", "dif_puntos_por_mano", "manos")


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def collect_entrants(args, rules: Rules) -> list[Entrant]:
    baselines = args.baselines
    if baselines is None:
        baselines = ["heuristic", "heuristic-mudo", "random"] if rules.num_players == 4 else ["heuristic", "random"]
    entrants: list[Entrant] = [parse_entrant(s) for s in args.specs]
    for run in args.runs:
        for path in run_checkpoints(run, per_run=args.per_run, iterations=args.iters):
            spec = f"ppo:{path.as_posix()}"
            entrants.append(Entrant(spec_label(spec), spec))
    entrants += [Entrant(spec_label(b), b) for b in baselines]
    unique: dict[str, Entrant] = {}
    for e in entrants:
        if e.name in unique:
            continue
        ok, reason = compatible(e.spec, rules)
        if not ok:
            print(f"salteo {e.name}: {reason}", flush=True)
            continue
        unique[e.name] = e
    return list(unique.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", nargs="*", default=[])
    parser.add_argument("--runs", nargs="*", default=[])
    parser.add_argument("--per-run", type=int, default=3)
    parser.add_argument("--iters", type=int, nargs="*", default=None)
    parser.add_argument("--baselines", nargs="*", default=None)
    parser.add_argument("--players", type=int, default=4, choices=(2, 4))
    parser.add_argument("--target", type=int, default=30)
    parser.add_argument("--spy", type=float, default=0.2)
    parser.add_argument("--no-signals", action="store_true")
    parser.add_argument("--matches", type=int, default=100)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--prior-sd", type=float, default=4.0)
    parser.add_argument("--anchor", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/liga")
    args = parser.parse_args()
    rules = Rules(num_players=args.players, target=args.target, signals=not args.no_signals and args.players == 4, spy_prob=args.spy)
    entrants = collect_entrants(args, rules)
    if len(entrants) < 2:
        raise SystemExit("la liga necesita al menos dos participantes compatibles")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = len(entrants) * (len(entrants) - 1) // 2
    print(f"{len(entrants)} participantes, {pairs} cruces de {args.matches} partidos", flush=True)
    start = time.time()

    def progress(done: int, total: int, result: dict) -> None:
        a, b = entrants[result["i"]].name, entrants[result["j"]].name
        print(f"[{done}/{total}] {a} vs {b}: {result['win_rate']:.3f} {result['ci95']}", flush=True)

    results = round_robin(rules, entrants, args.matches, seed=args.seed, workers=args.workers, progress=progress)
    summary = summarize_league(entrants, results, samples=args.bootstrap, seed=args.seed, prior_sd=args.prior_sd, anchor=args.anchor)
    summary["rules"] = rules.to_dict()
    summary["partidos_por_cruce"] = args.matches
    summary["seed"] = args.seed
    summary["segundos"] = round(time.time() - start, 1)
    (out / "liga.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(out / "ratings.csv", summary["ratings"], RATING_FIELDS)
    write_csv(out / "partidos.csv", summary["partidos"], PAIR_FIELDS)
    rows = summary["ratings"]
    anchor = summary["ancla"]
    title = f"Liga {rules.num_players} jugadores: Elo Bradley-Terry con IC 95% bootstrap"
    plot_intervals(
        [r["nombre"] for r in rows],
        [r["elo"] for r in rows],
        [r["elo_lo"] for r in rows],
        [r["elo_hi"] for r in rows],
        out / "liga.png",
        title,
        f"Elo relativo a {anchor}" if anchor else "Elo relativo a la media",
        reference=0.0,
        highlight=[r["nombre"] == anchor for r in rows],
        fmt="{:+.0f}",
    )
    names = summary["nombres"]
    order = [names.index(r["nombre"]) for r in rows]
    observed = [[summary["tasa_observada"][i][j] for j in order] for i in order]
    plot_matrix(
        [names[i] for i in order],
        [names[i] for i in order],
        [[float("nan") if v is None else v for v in row] for row in observed],
        out / "liga_matriz.png",
        "Tasa de victoria de la fila contra la columna",
        colorbar="victorias de la fila",
    )
    for r in rows:
        print(f"{r['nombre']:>28}  Elo {r['elo']:+7.1f}  [{r['elo_lo']:+7.1f}, {r['elo_hi']:+7.1f}]  p_mejor {r['p_mejor']:.2f}")
    print(f"resultados en {out}")


if __name__ == "__main__":
    main()
