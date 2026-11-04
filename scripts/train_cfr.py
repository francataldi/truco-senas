import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.arena import head_to_head
from agents.base import RandomAgent
from agents.cfr import BYTES_PER_INFOSET_BOUND, CFRAgent, infosets_for_budget, train_parallel
from agents.heuristic import HeuristicAgent
from truco.rules import Rules


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=300_000)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--target", type=int, default=30)
    parser.add_argument("--out", default="runs/cfr/heads_up.npz")
    parser.add_argument("--eval-matches", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--memory-gb", type=float, default=6.0)
    parser.add_argument("--max-infosets", type=int, default=None)
    parser.add_argument("--report-every", type=int, default=1000)
    parser.add_argument("--zero-score-prob", type=float, default=0.3)
    parser.add_argument("--regret-mode", choices=("linear", "plus", "vanilla"), default="linear")
    parser.add_argument("--keep-parts", action="store_true")
    args = parser.parse_args()
    rules = Rules(num_players=2, target=args.target, signals=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cap = infosets_for_budget(args.memory_gb, args.workers)
    if args.max_infosets is not None:
        cap = min(cap, args.max_infosets)
    print(
        f"{args.workers} workers x {args.iterations // max(1, args.workers)} iteraciones, "
        f"tope {cap} infosets por worker ({args.memory_gb:g} GB en total, cota {BYTES_PER_INFOSET_BOUND} B/infoset)",
        flush=True,
    )
    start = time.time()
    policy, curve = train_parallel(
        rules,
        args.iterations,
        args.workers,
        seed=args.seed,
        report_every=args.report_every,
        memory_gb=args.memory_gb,
        max_infosets=args.max_infosets,
        zero_score_prob=args.zero_score_prob,
        regret_mode=args.regret_mode,
        work_dir=out.parent / f"{out.stem}_partes",
        keep_parts=args.keep_parts,
    )
    elapsed = time.time() - start
    policy.meta["seconds"] = round(elapsed, 1)
    policy.save(out)
    curve_path = out.parent / "curva_infosets.csv"
    with open(curve_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(curve[0].keys()))
        writer.writeheader()
        writer.writerows(curve)
    print(f"{len(policy)} infosets en la política, {elapsed:.0f}s, {out.stat().st_size / 2**20:.1f} MB en disco", flush=True)
    agent = CFRAgent(policy, seed=args.seed)
    vs_heur = head_to_head(rules, [agent], [HeuristicAgent(seed=args.seed + 1)], args.eval_matches, seed=args.seed)
    vs_rand = head_to_head(rules, [agent], [RandomAgent(seed=args.seed + 2)], max(10, args.eval_matches // 4), seed=args.seed + 1)
    finals = {}
    for row in curve:
        finals[row["worker"]] = row
    summary = {
        "policy": str(out),
        "rules": rules.to_dict(),
        "abstraction": policy.abstraction.to_dict(),
        **policy.meta,
        "infosets": len(policy),
        "policy_bytes": policy.nbytes,
        "policy_bytes_per_infoset": round(policy.nbytes / max(1, len(policy)), 2),
        "file_bytes": out.stat().st_size,
        "training_bytes_per_infoset": [r["bytes_per_infoset"] for r in finals.values()],
        "peak_rss_mb_per_worker": [round(max(x["rss_bytes"] for x in curve if x["worker"] == w) / 2**20, 1) for w in finals],
        "vs_heuristic": vs_heur.to_dict(),
        "vs_random": vs_rand.to_dict(),
        "coverage": round(agent.coverage, 4),
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("CFR vs heuristico:", vs_heur.to_dict())
    print("CFR vs random:", vs_rand.to_dict())
    print(f"cobertura de infosets: {agent.coverage:.3f}")


if __name__ == "__main__":
    main()
