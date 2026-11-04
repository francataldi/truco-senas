import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.base import Agent
from agents.registry import make_agent
from truco import actions as A
from truco.cards import card_from_index, envido_points
from truco.engine import TrucoHand, TrucoMatch
from truco.rules import Rules

SEAT_NAMES = ("vos", "rival derecho", "compañero", "rival izquierdo")


class HumanAgent(Agent):
    name = "humano"

    def act(self, hand: TrucoHand, seat: int) -> int:
        legal = hand.legal_actions()
        show(hand, seat)
        for i, a in enumerate(legal):
            label = A.action_name(a, hand.next_truco_level() - 1)
            if a in A.PLAY_ACTIONS:
                label = f"jugar {card_from_index(hand.hands[seat][a])}"
            print(f"  [{i}] {label}")
        while True:
            raw = input("> ").strip()
            if raw.isdigit() and int(raw) < len(legal):
                return legal[int(raw)]
            print("elegí un número de la lista")


def show(hand: TrucoHand, seat: int) -> None:
    print()
    print(f"puntos: nosotros {hand.scores[0] + hand.points[0]} - ellos {hand.scores[1] + hand.points[1]}")
    cards = ", ".join(str(card_from_index(c)) for c in hand.remaining_cards(seat))
    print(f"tus cartas: {cards} (envido {envido_points(hand.hands[seat])})")
    for r in range(hand.round + 1):
        played = [f"{SEAT_NAMES[s]}: {card_from_index(hand.table[r][s])}" for s in range(hand.n) if hand.table[r][s] is not None]
        if played:
            print(f"mano {r + 1}: " + " | ".join(played))
    seen = [f"{SEAT_NAMES[s]} hizo la seña {hand.signal_view[seat][s]}" for s in range(hand.n) if s != seat and hand.signal_view[seat][s] is not None]
    if seen:
        print("señas: " + "; ".join(seen))
    if hand.pending is not None:
        what = A.TRUCO_NAMES[hand.pending.level - 1] if hand.pending.kind == "truco" else "envido"
        print(f"te cantaron {what}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partner", default="heuristic")
    parser.add_argument("--rivals", default="heuristic")
    parser.add_argument("--target", type=int, default=30)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    rules = Rules(num_players=4, target=args.target, signals=True)
    seats = [HumanAgent(), make_agent(args.rivals, seed=1), make_agent(args.partner, seed=2), make_agent(args.rivals, seed=3)]
    match = TrucoMatch(rules, rng=random.Random(args.seed))
    while not match.done:
        hand = match.new_hand()
        print(f"\n=== nueva mano, es mano: {SEAT_NAMES[hand.mano]} ===")
        while not hand.done:
            seat = hand.current_player
            action = seats[seat].act(hand, seat)
            if seat != 0 and not A.is_signal(action):
                label = A.action_name(action, hand.next_truco_level() - 1)
                if action in A.PLAY_ACTIONS:
                    label = f"juega {card_from_index(hand.hands[seat][action])}"
                print(f"{SEAT_NAMES[seat]}: {label}")
            hand.step(action)
        print(f"fin de la mano ({hand.end_reason}): nosotros +{hand.points[0]}, ellos +{hand.points[1]}")
        match.close_hand()
    print("\nganamos" if match.winner == 0 else "\nperdimos", match.scores)


if __name__ == "__main__":
    main()
