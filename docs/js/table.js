import { backSVG, cardSVG } from './cards.js';
import { esc, reducedMotion, sleep } from './dom.js';
import { faceSVG } from './faces.js';

const POS = ['south', 'east', 'north', 'west'];
const ROUND_LABELS = ['1ª', '2ª', '3ª'];

function rotation(card) {
  const x = Math.sin((card.index + 1) * 91.7) * 1000;
  return Math.round(((x - Math.floor(x)) * 12 - 6) * 10) / 10;
}

function trophySVG() {
  return `<svg class="trofeo" viewBox="0 0 64 64" aria-hidden="true">
    <path d="M18 8H46V22C46 31 40 37 32 37S18 31 18 22Z" fill="#f2c24f" stroke="#8a5d05" stroke-width="2"/>
    <path d="M18 12H9C9 21 13 26 19 27M46 12H55C55 21 51 26 45 27" fill="none" stroke="#8a5d05" stroke-width="2.5" stroke-linecap="round"/>
    <path d="M28 37H36V46H28Z" fill="#d49b1c"/>
    <path d="M20 54C20 48 24 46 32 46S44 48 44 54Z" fill="#f2c24f" stroke="#8a5d05" stroke-width="2"/>
    <path d="M25 14C25 22 27 27 31 30" fill="none" stroke="#fff3c4" stroke-width="2.5" stroke-linecap="round" opacity="0.8"/>
  </svg>`;
}

export class Table {
  constructor(root) {
    this.root = root;
    this.game = null;
    this.state = null;
    this.showAll = true;
    this.onPlay = null;
    this.signatures = new Map();
    this.locked = true;
  }

  setup(game) {
    this.game = game;
    this.viewer = game.viewer;
    this.signatures.clear();
    this.state = null;
    const seats = game.seats.map((seat) => this.seatMarkup(seat)).join('');
    const bazas = game.seats.map((seat) => `<div class="baza" data-seat="${seat.seat}" data-pos="${POS[seat.seat]}"></div>`).join('');
    const tools = [];
    if (game.mode === 'watch') {
      tools.push('<button type="button" class="felt-chip" data-action="toggle-cards"></button>');
    } else if (game.open_cards) {
      tools.push('<span class="felt-chip" title="Ves las cartas y la mente de los rivales">Cartas abiertas</span>');
    }
    if (game.signals) {
      tools.push(`<span class="felt-chip" title="Probabilidad de que un rival vea una seña">Pescar seña: ${Math.round(game.spy_prob * 100)}%</span>`);
    }
    this.root.innerHTML = `
      <div class="pano-info">
        <span class="hand-no" data-ref="hand-no">Mano 1</span>
        <div class="mini-score" data-ref="mini-score"></div>
      </div>
      <div class="pano-tools">${tools.join('')}</div>
      ${seats}
      <div class="centro">${bazas}<div class="tablero" data-ref="tablero"></div></div>
      <div class="overlay" data-ref="overlay" hidden></div>`;
    this.root.querySelector('[data-action="toggle-cards"]')?.addEventListener('click', () => {
      this.showAll = !this.showAll;
      this.updateToggle();
      if (this.state) this.render(this.state);
    });
    this.root.addEventListener('click', (event) => {
      const button = event.target.closest('.carta-btn.jugable');
      if (button && this.onPlay && !this.locked) this.onPlay(Number(button.dataset.action));
    });
    this.updateToggle();
  }

  updateToggle() {
    const button = this.root.querySelector('[data-action="toggle-cards"]');
    if (button) button.textContent = this.showAll ? 'Ocultar cartas' : 'Mostrar cartas';
  }

  seatMarkup(seat) {
    const own = this.viewer === seat.seat;
    return `<div class="asiento pos-${POS[seat.seat]} team-${seat.team}" data-seat="${seat.seat}">
      <div class="identidad">
        <div class="avatar" data-ref="avatar">${faceSVG(seat.seat)}</div>
        <div class="placa">
          <span class="rol">${esc(seat.role)}</span>
          <span class="agente" title="${esc(seat.description || '')}">${esc(seat.human ? 'Jugás vos' : seat.label)}</span>
          <span class="envido-chip" data-ref="envido"></span>
        </div>
        <span class="chip-mano" data-ref="mano" hidden>mano</span>
      </div>
      <div class="mano-cartas ${own ? 'mano-propia' : ''}" data-hand="${seat.seat}"></div>
    </div>`;
  }

  seatEl(seat) {
    return this.root.querySelector(`.asiento[data-seat="${seat}"]`);
  }

  setLocked(locked) {
    this.locked = locked;
    if (this.state) this.renderHands(this.state);
  }

  render(state, options = {}) {
    this.state = state;
    this.renderHands(state, options.deal);
    this.renderBazas(state);
    this.renderSeats(state);
    this.renderBoard(state);
  }

  renderSeats(state) {
    for (const seat of this.game.seats) {
      const el = this.seatEl(seat.seat);
      if (!el) continue;
      const active = state.turn === seat.seat && (state.phase === 'play' || state.phase === 'signals');
      el.classList.toggle('turno', active);
      el.querySelector('[data-ref="mano"]').hidden = state.mano !== seat.seat;
      const envido = state.seats[seat.seat].envido;
      const chip = el.querySelector('[data-ref="envido"]');
      const show = envido != null && (this.viewer === seat.seat || this.showAll || state.phase === 'hand_over' || state.phase === 'match_over');
      chip.textContent = show ? `envido ${envido}` : '';
    }
    const hand = this.root.querySelector('[data-ref="hand-no"]');
    if (hand) hand.textContent = `Mano ${state.hand}`;
    const mini = this.root.querySelector('[data-ref="mini-score"]');
    if (mini) {
      mini.innerHTML = this.game.teams
        .map((team, t) => `<span><i style="background:var(--team-${t})"></i><b>${state.scores[t]}</b>${esc(team.name)}</span>`)
        .join('');
    }
  }

  playableSlots(state) {
    const slots = new Map();
    if (this.viewer == null || state.turn !== this.viewer || this.locked) return slots;
    for (const option of state.legal) {
      if (option.kind === 'card') slots.set(option.slot, option.action);
    }
    return slots;
  }

  renderHands(state, deal = false) {
    for (const seat of this.game.seats) {
      const s = seat.seat;
      const container = this.root.querySelector(`[data-hand="${s}"]`);
      if (!container) continue;
      const info = state.seats[s];
      const own = this.viewer === s;
      const reveal = state.phase === 'hand_over' || state.phase === 'match_over';
      const open = own || ((this.showAll || reveal || this.viewer != null) && info.cards.some((c, i) => c && !info.played[i]));
      const slots = info.cards.map((card, i) => ({ card, i })).filter(({ i }) => !info.played[i]);
      const playable = own ? this.playableSlots(state) : new Map();
      const signature = `${state.hand}|${open}|${slots.map(({ card, i }) => `${i}:${card ? card.index : 'x'}:${playable.has(i) ? 1 : 0}`).join(',')}|${this.locked}`;
      if (!deal && this.signatures.get(s) === signature) continue;
      this.signatures.set(s, signature);
      container.classList.toggle('abierta', open && !own);
      container.classList.toggle('cerrada', !open);
      container.classList.toggle('espera', own && playable.size === 0);
      container.innerHTML = slots
        .map(({ card, i }, k) => {
          const dealt = deal ? ` reparte" style="--i:${k + s * 3}` : '';
          if (own && card) {
            const action = playable.get(i);
            const playableCls = action != null ? ' jugable' : '';
            const label = action != null ? `Jugar el ${card.name}` : card.name;
            const tab = action != null ? '' : ' tabindex="-1"';
            return `<div class="slot${dealt}" data-slot="${i}"><button type="button" class="carta-btn${playableCls}" data-action="${action ?? ''}" data-key="${k + 1}" aria-label="${esc(label)}"${tab}>${cardSVG(card)}</button></div>`;
          }
          const face = open && card && (this.showAll || own || reveal || this.viewer != null) ? cardSVG(card) : backSVG();
          return `<div class="slot${dealt}" data-slot="${i}">${face}</div>`;
        })
        .join('');
    }
  }

  renderBazas(state) {
    const current = state.round;
    for (const seat of this.game.seats) {
      const s = seat.seat;
      const container = this.root.querySelector(`.baza[data-seat="${s}"]`);
      if (!container) continue;
      const items = [];
      for (let r = 0; r < 3; r += 1) {
        const card = state.table[r][s];
        if (!card) continue;
        const result = state.rounds[r];
        let cls = '';
        if (result && !result.parda && result.seat === s) cls = ' gana';
        if (result && result.parda) {
          const top = Math.max(...state.table[r].filter(Boolean).map((c) => c.strength));
          if (card.strength === top) cls = ' parda';
        }
        if (r < current && state.phase === 'play') cls += ' vieja';
        items.push({ r, card, cls });
      }
      const signature = `${state.hand}|${items.map((it) => `${it.r}:${it.card.index}:${it.cls}`).join(',')}`;
      if (this.signatures.get(`b${s}`) === signature) continue;
      this.signatures.set(`b${s}`, signature);
      container.innerHTML = items
        .map(({ r, card, cls }) => {
          const rot = rotation(card);
          return `<div class="carta-mesa${cls}" data-round="${r}" data-rot="${rot}" style="--rot:${rot}deg">${cardSVG(card)}</div>`;
        })
        .join('');
    }
  }

  renderBoard(state) {
    const board = this.root.querySelector('[data-ref="tablero"]');
    if (!board) return;
    const rounds = ROUND_LABELS.map((label, r) => {
      const result = state.rounds[r];
      let cls = '';
      if (result) cls = result.parda ? ' parda' : ` t${result.winner}`;
      else if (r === state.round && state.phase === 'play') cls = ' actual';
      const title = result ? (result.parda ? 'parda' : `para ${this.game.teams[result.winner].name}`) : 'por jugar';
      return `<span class="ronda${cls}" title="${ROUND_LABELS[r]}: ${esc(title)}">${label}</span>`;
    }).join('');
    const parts = [`<div class="rondas">${rounds}</div>`];
    if (state.phase === 'signals') {
      parts.push('<div class="valor">Ronda de <b>señas</b></div>');
    } else if (state.truco.level > 0) {
      parts.push(`<div class="valor">${esc(state.truco.name[0].toUpperCase() + state.truco.name.slice(1))} querido · vale <b>${state.truco.value}</b></div>`);
    } else {
      parts.push(`<div class="valor">Se juega por <b>${state.truco.value}</b></div>`);
    }
    if (state.pending) {
      const name = state.pending.name;
      const who = this.game.seats[state.pending.responder].role;
      const verb = this.viewer === state.pending.responder ? 'respondés vos' : `responde ${who}`;
      parts.push(`<div class="canto-pendiente ${state.pending.kind}">¡${esc(name[0].toUpperCase() + name.slice(1))}! · ${esc(verb)}</div>`);
    }
    if (state.envido.done && state.envido.winner != null) {
      parts.push(`<div class="envido-res">Envido: ${esc(this.game.teams[state.envido.winner].name)} +${state.envido.value}</div>`);
    }
    board.innerHTML = parts.join('');
  }

  sourceRect(seat, slot) {
    const hand = this.root.querySelector(`[data-hand="${seat}"]`);
    if (!hand) return null;
    const exact = hand.querySelector(`.slot[data-slot="${slot}"]`);
    const target = exact && !hand.classList.contains('cerrada') ? exact : hand.querySelector('.slot:last-child');
    return target ? target.getBoundingClientRect() : null;
  }

  async flyCard(seat, round, from, duration) {
    const el = this.root.querySelector(`.baza[data-seat="${seat}"] .carta-mesa[data-round="${round}"]`);
    if (!el || !from || reducedMotion()) return;
    const to = el.getBoundingClientRect();
    const dx = from.left + from.width / 2 - (to.left + to.width / 2);
    const dy = from.top + from.height / 2 - (to.top + to.height / 2);
    const scale = from.width / Math.max(1, to.width);
    const rot = Number(el.dataset.rot || 0);
    el.style.zIndex = '5';
    const animation = el.animate(
      [
        { transform: `translate(${dx}px, ${dy}px) scale(${scale}) rotate(${rot * 3}deg)`, offset: 0 },
        { transform: `translate(${dx * 0.12}px, ${dy * 0.12 - 14}px) scale(1.08) rotate(${rot}deg)`, offset: 0.78 },
        { transform: `rotate(${rot}deg)`, offset: 1 },
      ],
      { duration, easing: 'cubic-bezier(.25,.8,.25,1)' },
    );
    try {
      await animation.finished;
    } catch (error) {
      return;
    } finally {
      el.style.zIndex = '';
    }
  }

  pulseWinner(seat, round) {
    const el = this.root.querySelector(`.baza[data-seat="${seat}"] .carta-mesa[data-round="${round}"]`);
    if (!el) return;
    el.classList.remove('pulso');
    void el.offsetWidth;
    el.classList.add('pulso');
  }

  bubble(seat, text, kind, duration = 2200) {
    const el = this.seatEl(seat);
    if (!el || !text) return;
    const anchor = el.querySelector('.identidad');
    anchor.querySelectorAll('.globo').forEach((node) => node.remove());
    const node = document.createElement('div');
    node.className = `globo ${kind}`;
    node.textContent = text;
    node.style.animationDuration = `${duration}ms`;
    anchor.append(node);
    setTimeout(() => node.remove(), duration + 50);
  }

  gesture(seat, token, duration = 1600) {
    const el = this.seatEl(seat);
    if (!el) return;
    const avatar = el.querySelector('[data-ref="avatar"]');
    avatar.innerHTML = faceSVG(seat, token);
    avatar.classList.remove('gesto');
    void avatar.offsetWidth;
    avatar.classList.add('gesto');
    clearTimeout(avatar._timer);
    avatar._timer = setTimeout(() => {
      avatar.innerHTML = faceSVG(seat);
      avatar.classList.remove('gesto');
    }, duration);
  }

  hideOverlay() {
    const overlay = this.root.querySelector('[data-ref="overlay"]');
    if (overlay) {
      overlay.hidden = true;
      overlay.innerHTML = '';
    }
  }

  summary({ eyebrow, title, detail, points, buttons = [], auto = 0, final = false, docked = false }) {
    const overlay = this.root.querySelector('[data-ref="overlay"]');
    if (!overlay) return Promise.resolve(null);
    const chips = points
      ? `<div class="puntos">${this.game.teams
          .map((team, t) => `<span><i style="background:var(--team-${t})"></i>${esc(team.name)} +${points[t]}</span>`)
          .join('')}</div>`
      : '';
    const actions = buttons.length
      ? `<div class="acciones">${buttons
          .map((b) => `<button type="button" class="btn ${b.primary ? 'btn-primary' : ''}" data-choice="${esc(b.id)}">${esc(b.label)}${b.key ? ` <kbd>${esc(b.key)}</kbd>` : ''}</button>`)
          .join('')}</div>`
      : '';
    const progress = auto > 0 ? `<div class="progreso"><i style="animation-duration:${auto}ms"></i></div>` : '';
    overlay.innerHTML = `<div class="resumen${final ? ' final' : ''}" role="dialog" aria-live="polite">
      ${final ? trophySVG() : ''}
      <div class="eyebrow">${esc(eyebrow || '')}</div>
      <h3>${esc(title)}</h3>
      ${detail ? `<p class="detalle">${esc(detail)}</p>` : ''}
      ${chips}${actions}${progress}
    </div>`;
    overlay.classList.toggle('abajo', docked);
    overlay.hidden = false;
    const first = overlay.querySelector('[data-choice]');
    if (first) first.focus({ preventScroll: true });
    return new Promise((resolve) => {
      let timer = null;
      const finish = (choice) => {
        clearTimeout(timer);
        overlay.removeEventListener('click', onClick);
        this.pendingSummary = null;
        resolve(choice);
      };
      const onClick = (event) => {
        const button = event.target.closest('[data-choice]');
        if (button) finish(button.dataset.choice);
      };
      overlay.addEventListener('click', onClick);
      if (auto > 0) timer = setTimeout(() => finish('auto'), auto);
      this.pendingSummary = finish;
    });
  }

  resolveSummary(choice) {
    if (this.pendingSummary) this.pendingSummary(choice);
  }
}

export { POS, sleep };
