import { esc, pct } from './dom.js';
import { faceSVG } from './faces.js';

export const SPEEDS = [0.5, 0.75, 1, 1.5, 2, 3, 4];

const ICONS = {
  pause: '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><rect x="6" y="5" width="4" height="14" rx="1.2"/><rect x="14" y="5" width="4" height="14" rx="1.2"/></svg>',
  play: '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M8 5.5v13a1 1 0 0 0 1.5.86l10.4-6.5a1 1 0 0 0 0-1.72L9.5 4.64A1 1 0 0 0 8 5.5Z"/></svg>',
  step: '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M5 6.2v11.6a1 1 0 0 0 1.55.83L15 13V18a1 1 0 0 0 2 0V6a1 1 0 0 0-2 0v5L6.55 5.37A1 1 0 0 0 5 6.2Z"/></svg>',
};

const KEYS = { 3: 'T', 4: 'E', 5: 'R', 6: 'F', 7: 'Q', 8: 'N', 9: 'M' };
const ORDER = { answer: 0, truco: 1, envido: 2, flor: 3, mazo: 4 };

function speedLabel(speed) {
  return `${String(speed).replace('.', ',')}×`;
}

export class Controls {
  constructor(root) {
    this.root = root;
    this.game = null;
    this.handlers = {};
    this.paused = false;
    this.root.addEventListener('click', (event) => {
      const button = event.target.closest('[data-act], [data-ctl]');
      if (!button || button.disabled) return;
      if (button.dataset.act != null) this.emit('action', Number(button.dataset.act));
      else this.emit(button.dataset.ctl);
    });
    this.root.addEventListener('input', (event) => {
      if (event.target.matches('[data-ctl="speed"]')) {
        const speed = SPEEDS[Number(event.target.value)];
        const label = this.root.querySelector('[data-ref="speed"]');
        if (label) label.textContent = speedLabel(speed);
        this.emit('speed', speed);
      }
    });
  }

  on(name, handler) {
    this.handlers[name] = handler;
  }

  emit(name, value) {
    const handler = this.handlers[name];
    if (handler) handler(value);
  }

  setup(game, speed) {
    this.game = game;
    this.paused = false;
    const index = Math.max(0, SPEEDS.indexOf(speed));
    const transport = game.mode === 'watch'
      ? `<button type="button" class="icon-btn" data-ctl="pause" data-ref="pause" aria-label="Pausar" title="Pausar (espacio)">${ICONS.pause}</button>
         <button type="button" class="icon-btn" data-ctl="step" data-ref="step" aria-label="Siguiente jugada" title="Siguiente jugada (→)" disabled>${ICONS.step}</button>`
      : '';
    this.root.classList.toggle('watch', game.mode === 'watch');
    this.root.innerHTML = `
      <div class="estado" aria-live="polite"><div class="titulo" data-ref="titulo"></div><div class="sub" data-ref="sub"></div></div>
      <div class="acciones" data-ref="acciones"></div>
      <div class="ritmo">
        ${transport}
        <label class="velocidad"><span>Velocidad <b data-ref="speed">${speedLabel(SPEEDS[index])}</b></span>
          <input type="range" min="0" max="${SPEEDS.length - 1}" step="1" value="${index}" data-ctl="speed" aria-label="Velocidad de los agentes"></label>
      </div>`;
  }

  setPaused(paused) {
    this.paused = paused;
    const pause = this.root.querySelector('[data-ref="pause"]');
    const step = this.root.querySelector('[data-ref="step"]');
    if (pause) {
      pause.innerHTML = paused ? ICONS.play : ICONS.pause;
      pause.setAttribute('aria-label', paused ? 'Seguir' : 'Pausar');
      pause.title = paused ? 'Seguir (espacio)' : 'Pausar (espacio)';
    }
    if (step) step.disabled = !paused;
  }

  status(title, sub = '', thinking = false) {
    const t = this.root.querySelector('[data-ref="titulo"]');
    const s = this.root.querySelector('[data-ref="sub"]');
    if (t) t.innerHTML = `${esc(title)}${thinking ? ' <span class="pensando" aria-hidden="true"><i></i><i></i><i></i></span>' : ''}`;
    if (s) s.textContent = sub;
  }

  clear() {
    const box = this.root.querySelector('[data-ref="acciones"]');
    if (box) box.innerHTML = '';
  }

  actions(state) {
    const box = this.root.querySelector('[data-ref="acciones"]');
    if (!box) return;
    const options = state.legal.filter((o) => o.kind !== 'card' && o.kind !== 'signal');
    options.sort((a, b) => (ORDER[a.kind] ?? 9) - (ORDER[b.kind] ?? 9) || a.action - b.action);
    const buttons = options.map((o) => {
      let cls = o.kind;
      if (o.kind === 'answer') cls = o.action === 7 ? 'quiero' : 'no-quiero';
      if (o.kind === 'flor') cls = 'envido';
      const title = o.kind === 'envido' && o.primero ? 'El envido está primero' : o.label;
      const key = KEYS[o.action];
      return `<button type="button" class="accion ${cls}" data-act="${o.action}" title="${esc(title)}">${esc(o.label)}${key ? ` <kbd>${key}</kbd>` : ''}</button>`;
    });
    const cards = state.legal.some((o) => o.kind === 'card');
    box.innerHTML = buttons.join('');
    if (cards && !buttons.length) box.insertAdjacentHTML('beforeend', '<span class="pista">Tocá una carta para jugarla</span>');
  }

  picker(state, suggested, partnerMute) {
    const box = this.root.querySelector('[data-ref="acciones"]');
    if (!box) return;
    const options = state.legal.filter((o) => o.kind === 'signal');
    const items = options.map((o, i) => {
      const meaning = partnerMute ? 'tu compañero no usa señas' : o.meaning || '';
      const isSuggested = suggested != null && suggested.token === o.token;
      const extra = isSuggested && suggested.p != null ? ` · la IA la elegiría con probabilidad ${pct(suggested.p)}` : '';
      return `<button type="button" class="gesto-btn${isSuggested ? ' sugerida' : ''}" data-act="${o.action}" style="--i:${i}" title="${esc(`${o.gesture}: ${meaning}${extra}`)}">
        <span class="mini-cara">${faceSVG(0, o.token)}</span>
        <span class="g-nombre">${i + 1}. ${esc(o.gesture[0].toUpperCase() + o.gesture.slice(1))}</span>
        <span class="g-sentido">${esc(meaning)}</span>
      </button>`;
    });
    box.innerHTML = `<div class="picker"><div class="picker-grid">${items.join('')}</div></div>`;
  }

  buttons(list) {
    const box = this.root.querySelector('[data-ref="acciones"]');
    if (!box) return;
    box.innerHTML = list
      .map((b) => `<button type="button" class="accion ${b.cls || 'siguiente'}" data-ctl="${b.ctl}">${esc(b.label)}${b.key ? ` <kbd>${esc(b.key)}</kbd>` : ''}</button>`)
      .join('');
  }
}
