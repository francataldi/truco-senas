from __future__ import annotations

import random
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

from agents.base import Agent
from truco import actions as A
from truco.cards import card_from_index
from truco.engine import PARDA, TRUCO_VALUES, TrucoMatch
from truco.rules import Rules

from .catalog import BUILTIN, HUMAN, AgentInfo, Catalog, SpecError
from .dictionary import Dictionary, DictionaryService, concept_holds, gesture
from .mind import Thought, advise, beliefs_payload, context, decide

PLAY_ROLES = ("Vos", "Rival derecho", "Compañero", "Rival izquierdo")
WATCH_ROLES = ("Sur", "Este", "Norte", "Oeste")
PLAY_TEAMS = ("Nosotros", "Ellos")
WATCH_TEAMS = ("Pareja A", "Pareja B")
ROUND_NAMES = ("primera", "segunda", "tercera")
ENVIDO_NAMES = {A.ENVIDO: "envido", A.REAL_ENVIDO: "real envido", A.FALTA_ENVIDO: "falta envido"}
TARGETS = (15, 30)


class GameError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass
class Seat:
    index: int
    role: str
    info: AgentInfo
    agent: Agent | None
    language: Dictionary
    value_scale: float

    @property
    def human(self) -> bool:
        return self.agent is None


def card_name(index: int) -> str:
    card = card_from_index(index)
    if card.rank == 1 and card.suit in ("espada", "basto"):
        return f"ancho de {card.suit}"
    return f"{card.rank} de {card.suit}"


def card_payload(index: int) -> dict:
    card = card_from_index(index)
    return {"index": index, "rank": card.rank, "suit": card.suit, "name": card_name(index), "strength": card.strength}


def truco_name(level: int) -> str:
    return A.TRUCO_NAMES[max(0, min(level, 3) - 1)]


class GameSession:
    def __init__(
        self,
        mode: str,
        seats: list[Seat],
        rules: Rules,
        seed: int | None = None,
        open_cards: bool = False,
        scramble: tuple[int, ...] = (),
    ) -> None:
        self.id = secrets.token_urlsafe(8)
        self.mode = mode
        self.seats = seats
        self.rules = rules
        self.open_cards = open_cards
        self.scramble = tuple(scramble)
        self.viewer = 0 if mode == "play" else None
        self.roles = PLAY_ROLES if mode == "play" else WATCH_ROLES
        self.teams = PLAY_TEAMS if mode == "play" else WATCH_TEAMS
        rng = random.Random(seed)
        self.match = TrucoMatch(rules, rng=random.Random(rng.random()), first_mano=rng.randrange(rules.num_players))
        self.hand = None
        self.hand_no = 0
        self.events: list[dict] = []
        self.seq = 0
        self.lock = threading.Lock()
        self.touched = time.monotonic()
        with self.lock:
            self._deal()
            if self.viewer is not None:
                self._run_agents(200)

    def describe(self) -> dict:
        return {
            "id": self.id,
            "mode": self.mode,
            "target": self.rules.target,
            "spy_prob": self.rules.spy_prob,
            "signals": self.rules.signals,
            "open_cards": self.open_cards,
            "scramble": list(self.scramble),
            "viewer": self.viewer,
            "teams": [self._team_payload(t) for t in (0, 1)],
            "seats": [
                {
                    "seat": s.index,
                    "role": s.role,
                    "team": s.index % 2,
                    "human": s.human,
                    **s.info.to_dict(),
                    "language": s.language.to_dict(),
                }
                for s in self.seats
            ],
        }

    def _team_payload(self, team: int) -> dict:
        labels = []
        for s in self.seats:
            if s.index % 2 == team and s.info.label not in labels:
                labels.append(s.info.label)
        return {"team": team, "name": self.teams[team], "agents": " y ".join(labels)}

    def since(self, seq: int) -> list[dict]:
        return self.events[max(0, seq) :]

    def state(self) -> dict:
        return self.events[-1]["state"]

    def advance(self, steps: int, expected_seq: int | None = None) -> list[dict]:
        with self.lock:
            self.touched = time.monotonic()
            self._check_seq(expected_seq)
            start = self.seq
            if self.hand.done:
                if self.match.done:
                    raise GameError(409, "la partida ya terminó")
                self._deal()
            self._run_agents(steps)
            return self.since(start)

    def act(self, action: int, expected_seq: int | None = None) -> list[dict]:
        with self.lock:
            self.touched = time.monotonic()
            self._check_seq(expected_seq)
            if self.viewer is None:
                raise GameError(409, "en modo espectador no hay jugador humano")
            hand = self.hand
            if hand.done:
                raise GameError(409, "la mano ya terminó, pedí la siguiente")
            if hand.current_player != self.viewer:
                raise GameError(409, "no es tu turno")
            legal = hand.legal_actions()
            if action not in legal:
                raise GameError(422, f"acción ilegal: {action} no está entre las legales {legal}")
            start = self.seq
            self._step(self.viewer, action, None)
            self._run_agents(200)
            return self.since(start)

    def advice(self) -> dict | None:
        with self.lock:
            self.touched = time.monotonic()
            hand = self.hand
            if self.viewer is None or hand.done or hand.current_player != self.viewer:
                return None
            advisor = self.seats[(self.viewer + 2) % 4]
            thought = advise(advisor.agent, hand, self.viewer, advisor.value_scale)
            payload = self._mind_payload(self.viewer, thought, force=True)
            payload["agent"] = BUILTIN["heuristic"][1] if advisor.info.family == "random" else advisor.info.label
            payload["advisor_seat"] = advisor.index
            return payload

    def _check_seq(self, expected: int | None) -> None:
        if expected is not None and expected != self.seq:
            raise GameError(409, f"la partida avanzó (evento {self.seq}), actualizá el estado")

    def _waiting_human(self) -> bool:
        return self.viewer is not None and self.hand.current_player == self.viewer

    def _run_agents(self, steps: int) -> None:
        for _ in range(steps):
            if self.hand.done or self._waiting_human():
                return
            seat = self.hand.current_player
            info = self.seats[seat]
            thought = decide(info.agent, self.hand, seat, info.value_scale)
            self._step(seat, thought.action, thought)

    def _deal(self) -> None:
        self.hand = self.match.new_hand(scramble_teams=self.scramble)
        self.hand_no += 1
        self.hidden_minds = []
        mano = self.hand.mano
        self._emit("deal", mano, f"Mano {self.hand_no}: es mano {self._who(mano)}")

    def _who(self, seat: int) -> str:
        return self.roles[seat]

    def _me(self, seat: int) -> bool:
        return self.viewer is not None and seat == self.viewer

    def _sees_cards(self, seat: int) -> bool:
        return self.viewer is None or self.open_cards or seat == self.viewer or self.hand.done

    def _sees_mind(self, seat: int) -> bool:
        return self.viewer is None or self.open_cards or seat % 2 == self.viewer % 2

    def _hides_mind(self, seat: int) -> bool:
        return not self._sees_cards(seat)

    def _visible_token(self, seat: int) -> int | None:
        hand = self.hand
        if self.viewer is None or self.open_cards or hand.done:
            return hand.signals_sent[seat]
        return hand.signal_view[self.viewer][seat]

    def label(self, seat: int, action: int) -> str:
        hand = self.hand
        if A.is_signal(action):
            token = A.signal_token(action)
            meaning = self.seats[seat].language.label(token)
            name = gesture(token)[0].capitalize()
            return f"{name} · {meaning}" if meaning else name
        if action in A.PLAY_ACTIONS:
            return card_name(hand.hands[seat][action])
        if action == A.TRUCO:
            name = truco_name(hand.next_truco_level())
            pending = hand.pending
            return f"Quiero {name}" if pending is not None and pending.kind == "truco" else name.capitalize()
        if action in ENVIDO_NAMES:
            return ENVIDO_NAMES[action].capitalize()
        return {A.QUIERO: "Quiero", A.NO_QUIERO: "No quiero", A.MAZO: "Irse al mazo", A.FLOR: "Flor"}[action]

    def _legal_payload(self, seat: int) -> list[dict]:
        hand = self.hand
        pending = hand.pending
        out = []
        for action in hand.legal_actions():
            item = {"action": action, "label": self.label(seat, action)}
            if A.is_signal(action):
                token = A.signal_token(action)
                item.update(kind="signal", token=token, gesture=gesture(token)[0], meaning=self.seats[seat].language.label(token))
            elif action in A.PLAY_ACTIONS:
                item.update(kind="card", slot=action, card=card_payload(hand.hands[seat][action]))
            elif action == A.TRUCO:
                item.update(kind="truco")
            elif action in ENVIDO_NAMES:
                item.update(kind="envido", primero=pending is not None and pending.kind == "truco")
            elif action in (A.QUIERO, A.NO_QUIERO):
                item.update(kind="answer")
            elif action == A.MAZO:
                item.update(kind="mazo")
            else:
                item.update(kind="flor")
            out.append(item)
        return out

    def _mind_payload(self, seat: int, thought: Thought | None, force: bool = False) -> dict | None:
        if thought is None or not (force or self._sees_mind(seat)):
            return None
        options = None
        if thought.probs is not None:
            ranked = sorted(thought.probs.items(), key=lambda kv: (-kv[1], kv[0]))
            options = [{"action": a, "label": self.label(seat, a), "p": round(p, 4)} for a, p in ranked]
        payload = {
            "kind": thought.kind,
            "seat": seat,
            "agent": self.seats[seat].info.label,
            "context": context(self.hand, you=force),
            "chosen": thought.action,
            "chosen_label": self.label(seat, thought.action),
            "options": options,
            "value": None if thought.value is None else round(thought.value, 3),
            "beliefs": beliefs_payload(thought.beliefs),
        }
        if force or not self._hides_mind(seat) or (options is None and payload["value"] is None):
            return payload
        payload["ref"] = len(self.hidden_minds)
        self.hidden_minds.append(payload)
        return {**payload, "options": None, "value": None, "hidden": True}

    def _step(self, seat: int, action: int, thought: Thought | None) -> None:
        hand = self.hand
        pending = hand.pending
        rounds_before = len(hand.round_winners)
        envido_before = hand.envido_done
        round_before = hand.round
        signal_phase = hand.signal_phase
        mind = self._mind_payload(seat, thought)
        say = None
        if action == A.TRUCO:
            name = truco_name(hand.next_truco_level())
            say = f"¡Quiero {name}!" if pending is not None and pending.kind == "truco" else f"¡{name.capitalize()}!"
        hand.step(action)
        who = self._who(seat)
        if signal_phase:
            token = self._visible_token(seat)
            spied = [o for o in range(hand.n) if o % 2 != seat % 2 and hand.signal_view[o][seat] is not None]
            if token is None:
                text = f"{who} le hizo una seña a su compañero"
            elif self._me(seat):
                text = f"Le hiciste una seña a tu compañero: {gesture(token)[2]}"
            else:
                text = f"{who} {gesture(token)[1]}"
            visible_spies = spied if self.viewer is None or self.open_cards else [o for o in spied if o == self.viewer]
            self._emit("signal", seat, text, token=token, spied_by=visible_spies, thought=mind)
        elif action in A.PLAY_ACTIONS:
            card = hand.table[round_before][seat]
            text = f"Jugaste el {card_name(card)}" if self._me(seat) else f"{who} jugó el {card_name(card)}"
            self._emit("play", seat, text, card=card_payload(card), slot=action, round=round_before, thought=mind)
            if len(hand.round_winners) > rounds_before:
                self._emit_round(rounds_before)
        elif action == A.TRUCO:
            self._emit("call", seat, f"{who}: {say}", say=say, call="truco", level=hand.pending.level if hand.pending else None, thought=mind)
        elif action in ENVIDO_NAMES:
            name = ENVIDO_NAMES[action]
            say = f"¡{name.capitalize()}!"
            if pending is not None and pending.kind == "truco":
                say = f"¡El envido está primero! {name.capitalize()}"
            self._emit("call", seat, f"{who}: {say}", say=say, call=name, thought=mind)
        elif action in (A.QUIERO, A.NO_QUIERO):
            accepted = action == A.QUIERO
            say = "¡Quiero!" if accepted else "No quiero"
            self._emit("answer", seat, f"{who}: {say}", say=say, accepted=accepted, about=pending.kind, thought=mind)
            if pending.kind == "envido" and hand.envido_done and not envido_before:
                self._emit_envido(accepted)
        elif action == A.MAZO:
            text = "Te fuiste al mazo" if self._me(seat) else f"{who} se fue al mazo"
            self._emit("mazo", seat, text, say="Me voy al mazo", thought=mind)
        elif action == A.FLOR:
            self._emit("call", seat, f"{who}: ¡Flor!", say="¡Flor!", call="flor", thought=mind)
        if hand.done:
            self._finish_hand(seat)

    def _emit_round(self, r: int) -> None:
        hand = self.hand
        winner = hand.round_winners[r]
        seat = hand.round_winner_seat[r]
        if winner == PARDA:
            self._emit("round", None, f"{ROUND_NAMES[r].capitalize()} parda", round=r, winner=None, parda=True)
            return
        card = hand.table[r][seat]
        text = f"{ROUND_NAMES[r].capitalize()} para {self.teams[winner]}: {self._who(seat)} con el {card_name(card)}"
        self._emit("round", seat, text, round=r, winner=winner, parda=False, card=card_payload(card))

    def _emit_envido(self, accepted: bool) -> None:
        hand = self.hand
        team = hand.envido_winner
        value = hand.envido_value
        if not accepted:
            self._emit("envido", None, f"Envido no querido: +{value} para {self.teams[team]}", winner=team, value=value, accepted=False, shown={})
            return
        winner_seat = next(s for s in range(hand.n) if hand.revealed_envido[s] is not None)
        shown = {winner_seat: hand.envido_of[winner_seat]}
        for s in range(hand.n):
            if self.viewer is None or self.open_cards or s == self.viewer:
                shown[s] = hand.envido_of[s]
        text = f"Envido: {self._who(winner_seat)} tenía {hand.envido_of[winner_seat]}. +{value} para {self.teams[team]}"
        self._emit("envido", winner_seat, text, winner=team, value=value, accepted=True, shown={str(k): v for k, v in shown.items()})

    def _finish_hand(self, last_seat: int) -> None:
        hand = self.hand
        team = hand.winner_team
        reason = hand.end_reason or ""
        value = TRUCO_VALUES[hand.truco_level]
        name = self.teams[team]
        if reason.startswith("target"):
            text = self._reached(team)
        elif reason == "mazo":
            text = f"{'Te fuiste' if self._me(last_seat) else self._who(last_seat) + ' se fue'} al mazo: +{value} para {name}"
        elif reason == "no_quiero":
            text = f"No quisieron el {truco_name(hand.truco_level + 1)}: +{value} para {name}"
        else:
            extra = f" con {truco_name(hand.truco_level)} querido" if hand.truco_level else ""
            text = f"Mano para {name}{extra}: +{value}"
        self.match.close_hand()
        checks = {}
        for s in range(hand.n):
            token = hand.signals_sent[s]
            if token is None:
                continue
            label = self.seats[s].language.label(token)
            checks[str(s)] = {"token": token, "label": label, "holds": concept_holds(label, hand.hands[s])}
        self._emit("hand_end", None, text, winner=team, reason=reason, points=list(hand.points), checks=checks, minds=self.hidden_minds)
        if self.match.done:
            w = self.match.winner
            scores = self.match.scores
            self._emit("match_end", None, self._won(w, scores[w], scores[1 - w]), winner=w, scores=list(scores))

    def _reached(self, team: int) -> str:
        target = self.rules.target
        if self.mode == "play":
            return f"¡Llegamos a {target}!" if team == 0 else f"Ellos llegaron a {target}"
        return f"¡La {self.teams[team]} llegó a {target}!"

    def _won(self, team: int, high: int, low: int) -> str:
        if self.mode == "play":
            return f"¡Ganamos {high} a {low}!" if team == 0 else f"Ganaron ellos, {high} a {low}"
        return f"¡Ganó la {self.teams[team]}, {high} a {low}!"

    def _emit(self, kind: str, seat: int | None, text: str, **extra) -> None:
        self.seq += 1
        event = {"seq": self.seq, "type": kind, "seat": seat, "text": text, "thought": None, **extra}
        event["state"] = self._snapshot()
        self.events.append(event)

    def _snapshot(self) -> dict:
        hand = self.hand
        seats = []
        for s in range(hand.n):
            visible = self._sees_cards(s)
            cards = [card_payload(c) if visible or hand.played_slot[s][i] else None for i, c in enumerate(hand.hands[s])]
            seats.append(
                {
                    "seat": s,
                    "team": s % 2,
                    "cards": cards,
                    "played": list(hand.played_slot[s]),
                    "envido": hand.envido_of[s] if visible else None,
                }
            )
        pending = None
        if hand.pending is not None:
            p = hand.pending
            if p.kind == "truco":
                name = truco_name(p.level)
            else:
                name = ", ".join(ENVIDO_NAMES[c] for c in hand.envido_calls)
            pending = {"kind": p.kind, "caller": p.caller, "responder": p.responder, "level": p.level, "name": name}
        signals = []
        for s in range(hand.n):
            token = self._visible_token(s)
            signals.append({"seat": s, "sent": hand.signals_sent[s] is not None, "token": token})
        if hand.done:
            phase = "match_over" if self.match.done else "hand_over"
        else:
            phase = "signals" if hand.signal_phase else "play"
        turn = hand.current_player
        result = None
        if hand.done:
            result = {"winner": hand.winner_team, "reason": hand.end_reason, "points": list(hand.points)}
        return {
            "hand": self.hand_no,
            "phase": phase,
            "mano": hand.mano,
            "turn": turn,
            "scores": [hand.scores[t] + hand.points[t] for t in (0, 1)],
            "points": list(hand.points),
            "target": self.rules.target,
            "truco": {
                "level": hand.truco_level,
                "holder": hand.truco_holder,
                "value": TRUCO_VALUES[hand.truco_level],
                "name": truco_name(hand.truco_level) if hand.truco_level else None,
            },
            "pending": pending,
            "envido": {
                "calls": [ENVIDO_NAMES[c] for c in hand.envido_calls],
                "done": hand.envido_done,
                "winner": hand.envido_winner,
                "value": hand.envido_value,
            },
            "round": min(hand.round, 2),
            "rounds": [
                {"winner": None if w == PARDA else w, "parda": w == PARDA, "seat": hand.round_winner_seat[i]}
                for i, w in enumerate(hand.round_winners)
            ],
            "table": [[None if c is None else card_payload(c) for c in row] for row in hand.table],
            "seats": seats,
            "signals": signals,
            "legal": self._legal_payload(turn) if self.viewer is not None and turn == self.viewer else [],
            "result": result,
            "match": {"done": self.match.done, "winner": self.match.winner, "scores": list(self.match.scores)},
        }


class GameFactory:
    def __init__(self, catalog: Catalog, dictionaries: DictionaryService) -> None:
        self.catalog = catalog
        self.dictionaries = dictionaries

    def create(
        self,
        mode: str,
        specs: list[str],
        target: int = 30,
        seed: int | None = None,
        open_cards: bool = False,
        spy_prob: float = 0.2,
        scramble: tuple[int, ...] = (),
    ) -> GameSession:
        if mode not in ("play", "watch"):
            raise GameError(422, f"modo desconocido: {mode}")
        if len(specs) != 4:
            raise GameError(422, "hacen falta cuatro asientos")
        if target not in TARGETS:
            raise GameError(422, f"la partida se juega a {' o '.join(map(str, TARGETS))}")
        if any(t not in (0, 1) for t in scramble):
            raise GameError(422, "las parejas son 0 y 1")
        try:
            infos = [self.catalog.resolve(spec) for spec in specs]
        except SpecError as exc:
            raise GameError(422, str(exc)) from exc
        humans = [i for i, info in enumerate(infos) if info.kind == HUMAN]
        if mode == "play" and humans != [0]:
            raise GameError(422, "para jugar, el asiento 0 es tuyo y los otros tres son agentes")
        if mode == "watch" and humans:
            raise GameError(422, "en modo espectador juegan cuatro agentes")
        vocabs = {info.details.get("signal_vocab", 8) for info in infos if info.family == "ppo"}
        if len(vocabs) > 1:
            raise GameError(422, "los agentes usan vocabularios de señas distintos")
        vocab = vocabs.pop() if vocabs else 8
        signals = not all(info.mute for info in infos)
        rules = Rules(num_players=4, target=target, flor=False, signals=signals, signal_vocab=vocab, spy_prob=spy_prob)
        rng = random.Random(seed)
        roles = PLAY_ROLES if mode == "play" else WATCH_ROLES
        languages = [None if info.kind == HUMAN else self.dictionaries.get(info) for info in infos]
        if mode == "play":
            languages[0] = languages[2]
        seats = []
        for i, info in enumerate(infos):
            agent = None if info.kind == HUMAN else self.catalog.make(info, seed=rng.randrange(2**31))
            scale = self.catalog.reward_scale(info)
            seats.append(Seat(i, roles[i], info, agent, languages[i], scale))
        return GameSession(mode, seats, rules, seed=rng.randrange(2**31), open_cards=open_cards, scramble=scramble)


class GameStore:
    def __init__(self, limit: int = 64) -> None:
        self.limit = limit
        self._games: OrderedDict[str, GameSession] = OrderedDict()
        self._lock = threading.Lock()

    def add(self, game: GameSession) -> None:
        with self._lock:
            self._games[game.id] = game
            while len(self._games) > self.limit:
                self._games.popitem(last=False)

    def get(self, game_id: str) -> GameSession:
        with self._lock:
            game = self._games.get(game_id)
            if game is None:
                raise GameError(404, "no existe esa partida")
            self._games.move_to_end(game_id)
            return game

    def remove(self, game_id: str) -> None:
        with self._lock:
            if self._games.pop(game_id, None) is None:
                raise GameError(404, "no existe esa partida")

    def __len__(self) -> int:
        return len(self._games)
