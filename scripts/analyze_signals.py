import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.arena import head_to_head
from agents.heuristic import HeuristicAgent
from agents.ppo.agent import PPOAgent
from agents.ppo.net import load_checkpoint
from agents.signal_analysis import collect_signals, plot_dictionary, summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hands", type=int, default=20_000)
    parser.add_argument("--matches", type=int, default=400)
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    net, rules, _ = load_checkpoint(args.checkpoint)
    out = Path(args.out) if args.out else Path(args.checkpoint).parent / "senas"
    out.mkdir(parents=True, exist_ok=True)
    agent = PPOAgent(net, seed=args.seed)
    data = collect_signals(rules, [agent] * rules.num_players, args.hands, seed=args.seed)
    summary = summarize(data, rules.signal_vocab)
    full = head_to_head(rules, [agent], [agent], args.matches, seed=args.seed + 1, scramble_b=True)
    heur = head_to_head(rules, [agent], [HeuristicAgent(seed=args.seed)], args.matches, seed=args.seed + 2)
    classic = head_to_head(rules, [HeuristicAgent(seed=1)], [HeuristicAgent(seed=2, use_signals=False)], args.matches, seed=args.seed + 3)
    summary["valor_de_las_señas"] = {
        "ppo_vs_ppo_con_señas_mezcladas": full.to_dict(),
        "ppo_vs_heuristico": heur.to_dict(),
        "heuristico_con_señas_clasicas_vs_sin_señas": classic.to_dict(),
    }
    (out / "diccionario.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_dictionary(summary, str(out / "diccionario.png"))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("p_rasgo_dada_seña", "p_rasgo_base")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
