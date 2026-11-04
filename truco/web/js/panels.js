import { clamp, esc, pct, signed } from './dom.js';
import { faceSVG, GESTURES } from './faces.js';

const IMPORTANT = new Set(['call', 'envido', 'mazo', 'hand_end', 'match_end']);

export function actionKind(action) {
  if (action >= 11) return 'signal';
  if (action <= 2) return 'card';
  if (action === 3) return 'truco';
  if (action >= 4 && action <= 6) return 'envido';
  if (action === 7) return 'quiero';
  if (action === 8) return 'no-quiero';
  if (action === 9) return 'mazo';
  return 'otra';
}

function miniFace(seat, token = null) {
  return `<span class="mini-cara">${faceSVG(seat, token)}</span>`;
}

function capital(text) {
  return text ? text[0].toUpperCase() + text.slice(1) : '';
}

export class Log {
  constructor(root, hint) {
    this.root = root;
    this.hint = hint;
  }

  clear() {
    this.root.innerHTML = '';
    if (this.hint) this.hint.textContent = '';
  }

  add(event, game) {
    const nearBottom = this.root.scrollHeight - this.root.scrollTop - this.root.clientHeight < 60;
    const li = document.createElement('li');
    if (event.type === 'deal') {
      li.className = 'mano-sep';
      li.textContent = `Mano ${event.state.hand}`;
      this.root.append(li);
      if (this.hint) this.hint.textContent = `mano ${event.state.hand}`;
    } else {
      const team = event.seat != null ? game.seats[event.seat].team : null;
      const classes = [];
      if (team != null) classes.push(`team-${team}`);
      if (IMPORTANT.has(event.type)) classes.push('importante');
      if (event.type === 'hand_end' || event.type === 'match_end') classes.push('resultado');
      li.className = classes.join(' ');
      li.innerHTML = `<span class="punto"></span><span>${esc(event.text)}</span>`;
      this.root.append(li);
    }
    while (this.root.children.length > 500) this.root.firstElementChild.remove();
    if (nearBottom || event.type === 'deal') this.root.scrollTop = this.root.scrollHeight;
  }
}

export class SignalsPanel {
  constructor(root, hint) {
    this.root = root;
    this.hint = hint;
    this.game = null;
    this.state = null;
    this.checks = null;
    this.spied = new Set();
    this.language = null;
    this.open = null;
    this.root.addEventListener('click', (event) => {
      const tab = event.target.closest('[data-lang]');
      if (tab) {
        this.language = tab.dataset.lang;
        this.open = null;
        this.renderDictionary();
        return;
      }
      const item = event.target.closest('[data-token]');
      if (item) {
        const token = Number(item.dataset.token);
        this.open = this.open === token ? null : token;
        this.renderDictionary();
      }
    });
  }

  languages() {
    const seen = new Map();
    const order = this.game.viewer != null ? [2, 1, 3] : [0, 1, 2, 3];
    for (const s of order) {
      const seat = this.game.seats[s];
      if (seat.human) continue;
      if (!seen.has(seat.spec)) seen.set(seat.spec, { spec: seat.spec, label: seat.label, language: seat.language, seats: [s] });
      else seen.get(seat.spec).seats.push(s);
    }
    return [...seen.values()];
  }

  setup(game) {
    this.game = game;
    this.state = null;
    this.checks = null;
    this.spied = new Set();
    const languages = this.languages();
    this.language = languages.length ? languages[0].spec : null;
    this.open = null;
    this.root.innerHTML = '<ul class="senas-lista" data-ref="lista"></ul><div class="diccionario" data-ref="dic"></div>';
    if (this.hint) this.hint.textContent = game.signals ? 'de esta mano' : 'nadie usa señas';
    this.renderDictionary();
  }

  newHand() {
    this.checks = null;
    this.spied = new Set();
  }

  update(state, event = null) {
    this.state = state;
    if (event && event.type === 'signal' && event.spied_by && event.spied_by.length) this.spied.add(event.seat);
    if (event && event.type === 'hand_end') this.checks = event.checks || null;
    this.renderRows(event && event.type === 'signal' ? event.seat : null);
  }

  meaning(seat, token) {
    const language = this.game.seats[seat].language;
    if (!language || token == null) return null;
    return language.tokens[token] || null;
  }

  renderRows(flash) {
    const list = this.root.querySelector('[data-ref="lista"]');
    if (!list || !this.state) return;
    const viewer = this.game.viewer;
    const order = viewer != null ? [0, 2, 1, 3] : [0, 2, 1, 3];
    const rows = order.map((s) => {
      const seat = this.game.seats[s];
      const signal = this.state.signals[s];
      const token = signal.token;
      let face = miniFace(s);
      let who = `<b>${esc(seat.role)}</b>`;
      let meaning = '';
      let badge = '';
      if (!this.game.signals) {
        meaning = '<div class="significado tenue">sin ronda de señas</div>';
      } else if (!signal.sent) {
        meaning = `<div class="significado tenue">${this.state.phase === 'signals' ? 'todavía no hizo su seña' : '—'}</div>`;
      } else if (token == null) {
        meaning = '<div class="significado tenue">hizo una seña que no viste</div>';
        badge = '<span class="badge" title="Sólo la ve su compañero, salvo que la pesques">oculta</span>';
      } else {
        face = miniFace(s, token);
        const entry = this.meaning(s, token);
        const mute = seat.language && seat.language.source === 'mudo';
        who += ` · ${esc(GESTURES[token] || `seña ${token}`)}`;
        const label = mute ? 'no usa señas' : entry ? entry.label : '';
        const dim = mute || !entry || entry.confidence == null;
        meaning = `<div class="significado${dim ? ' tenue' : ''}" title="${esc(label)}">${esc(label)}</div>`;
        if (this.checks && this.checks[s] && this.checks[s].holds != null) {
          badge = this.checks[s].holds
            ? '<span class="badge ok" title="Su mano coincidía con lo que anunció">✓ cumplió</span>'
            : '<span class="badge mal" title="Su mano no coincidía con lo que anunció">✗ no cumplió</span>';
        } else if (viewer != null && s % 2 !== viewer % 2 && this.spied.has(s)) {
          badge = '<span class="badge oro" title="Le viste la seña al rival">¡pescada!</span>';
        } else if (entry && entry.confidence != null) {
          badge = `<span class="badge azul" title="Probabilidad de que la seña diga la verdad según el diccionario">${pct(entry.confidence)}</span>`;
        }
        if (viewer != null && s === viewer) who = `<b>Vos</b> · ${esc(GESTURES[token] || '')}`;
      }
      return `<li class="sena-row team-${seat.team}${flash === s ? ' flash' : ''}" data-seat="${s}">
        ${face}
        <div class="sena-texto"><div class="quien">${who}</div>${meaning}</div>
        ${badge}
      </li>`;
    });
    list.innerHTML = rows.join('');
  }

  renderDictionary() {
    const box = this.root.querySelector('[data-ref="dic"]');
    if (!box || !this.game) return;
    const languages = this.languages();
    const current = languages.find((l) => l.spec === this.language) || languages[0];
    if (!current) {
      box.innerHTML = '';
      return;
    }
    const viewer = this.game.viewer;
    const title = viewer != null && current.seats.includes(2) ? 'Idioma de tu compañero' : `Idioma de ${current.label}`;
    const tabs = languages.length > 1
      ? `<div class="dic-tabs" role="group" aria-label="Diccionarios">${languages
          .map((l) => {
            const name = viewer != null ? (l.seats.includes(2) ? 'Compañero' : 'Rivales') : this.teamName(l.seats);
            return `<button type="button" data-lang="${esc(l.spec)}" aria-pressed="${l.spec === current.spec}" title="${esc(l.label)}">${esc(name)}</button>`;
          })
          .join('')}</div>`
      : '';
    const lang = current.language;
    let meta = '';
    if (lang.source === 'archivo') meta = `Aprendido por la red · diccionario.json con ${lang.hands.toLocaleString('es-AR')} señas`;
    else if (lang.source === 'calculado') meta = `Calculado al arrancar sobre ${lang.hands.toLocaleString('es-AR')} señas`;
    else meta = 'Este agente no usa señas: siempre hace el mismo gesto';
    if (lang.mi_bits != null && lang.source !== 'mudo') meta += ` · informa ${String(lang.mi_bits.toFixed(2)).replace('.', ',')} bits sobre su mejor carta`;
    const items = lang.tokens.map((t) => {
      const unused = t.frequency === 0;
      const dim = t.confidence == null;
      const bar = t.confidence != null ? Math.round(t.confidence * 100) : 0;
      const profile = this.open === t.token && !unused ? this.profile(t) : '';
      return `<li class="dic-item${unused ? ' sin-uso' : ''}${profile ? ' abierto' : ''}">
        <button type="button" data-token="${t.token}" aria-expanded="${profile ? 'true' : 'false'}">
          ${miniFace(current.seats[0], t.token)}
          <span><span class="nombre">${esc(capital(t.gesture))} · se usa ${pct(t.frequency)}</span><span class="sentido${dim ? ' tenue' : ''}">${esc(t.label)}</span></span>
          <span class="medidor"><span>${t.confidence != null ? pct(t.confidence) : '—'}</span><span class="barra"><i style="width:${bar}%"></i></span></span>
        </button>
        ${profile}
      </li>`;
    });
    box.innerHTML = `<div class="dic-head"><h3>${esc(title)}</h3>${tabs}</div>
      <div class="dic-meta">${esc(meta)}</div>
      <ul class="dic-lista">${items.join('')}</ul>`;
  }

  teamName(seats) {
    const team = this.game.seats[seats[0]].team;
    return this.game.teams[team].name;
  }

  profile(token) {
    if (!token.features || !token.features.length) return '';
    const rows = token.features.map((f) => {
      const width = clamp(f.p * 100, 0, 100);
      const base = clamp(f.base * 100, 0, 100);
      return `<span class="rasgo" title="${esc(f.label)}">${esc(f.label)}</span>
        <span class="barra-base" title="Base ${pct(f.base)}"><i style="width:${width}%"></i><b style="left:${base}%"></b></span>
        <span class="pct">${pct(f.p)}</span>`;
    });
    return `<div class="perfil">${rows.join('')}<span class="perfil-nota">Probabilidad de cada rasgo de la mano cuando hace esta seña. La rayita marca la frecuencia normal.</span></div>`;
  }
}

export class MindPanel {
  constructor(root, toggle) {
    this.root = root;
    this.toggle = toggle;
    this.enabled = toggle ? toggle.checked : true;
    this.game = null;
    this.thoughts = [];
    this.selected = null;
    this.advice = null;
    this.onToggle = null;
    if (toggle) {
      toggle.addEventListener('change', () => {
        this.enabled = toggle.checked;
        this.render();
        if (this.onToggle) this.onToggle(this.enabled);
      });
    }
    this.root.addEventListener('click', (event) => {
      const pick = event.target.closest('[data-pick]');
      if (pick) {
        const value = pick.dataset.pick;
        this.selected = value === 'seguir' ? null : Number(value);
        this.render();
      }
    });
  }

  setup(game) {
    this.game = game;
    this.thoughts = [];
    this.selected = null;
    this.advice = null;
    this.render();
  }

  newHand() {
    this.thoughts = [];
    this.selected = null;
    this.advice = null;
    this.render();
  }

  push(event) {
    if (!event.thought) return;
    this.thoughts.push(event.thought);
    this.advice = null;
    if (this.enabled) this.render();
  }

  setAdvice(advice) {
    this.advice = advice;
    if (this.enabled) this.render();
  }

  clearAdvice() {
    if (this.advice) {
      this.advice = null;
      if (this.enabled) this.render();
    }
  }

  render() {
    if (!this.game) {
      this.root.innerHTML = this.empty('Empezá una partida para ver cómo decide cada agente.');
      return;
    }
    if (!this.enabled) {
      this.root.innerHTML = this.empty('Activala para ver, en cada decisión, las probabilidades de la política, el valor estimado y lo que cree cada agente sobre su compañero.');
      return;
    }
    const showingAdvice = this.advice && this.selected == null;
    const thought = showingAdvice ? this.advice : this.selected != null ? this.thoughts[this.selected] : this.thoughts[this.thoughts.length - 1];
    let body = '';
    if (!thought) {
      const hidden = this.game.viewer != null && !this.game.open_cards;
      body = this.empty(hidden ? 'Todavía no decidió nadie de tu equipo en esta mano. La mente de los rivales queda oculta salvo que juegues con cartas abiertas.' : 'Todavía no hubo decisiones en esta mano.');
    } else {
      body = this.card(thought, showingAdvice);
    }
    this.root.innerHTML = body + this.timeline();
  }

  empty(text) {
    return `<div class="mente-vacia">
      <svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true">
        <path d="M18 40v-6c-5-2-9-7-9-13a15 15 0 0 1 30 0c0 3-1 6-3 8l2 6-6-1c-2 1-4 2-6 2v4"/>
        <path d="M20 18a4 4 0 0 1 8 0c0 3-4 3-4 6M24 29v.5"/>
      </svg>
      <div>${esc(text)}</div>
    </div>`;
  }

  card(thought, advice) {
    const seat = this.game.seats[thought.seat];
    const name = advice ? 'Tu turno' : seat.role;
    const sub = advice ? `Qué haría ${thought.agent} en tu lugar` : thought.agent;
    const parts = [];
    parts.push(`<div class="mente-quien">${miniFace(thought.seat)}
      <div style="min-width:0"><div class="nombre">${esc(name)} <span class="mente-contexto">${esc(thought.context)}</span></div><div class="sub" title="${esc(sub)}">${esc(sub)}</div></div></div>`);
    if (thought.value != null) {
      const v = thought.value;
      const pos = clamp(50 + (v / 4) * 50, 2, 98);
      const tone = v > 0.05 ? 'pos' : v < -0.05 ? 'neg' : '';
      parts.push(`<div title="Diferencia de puntos que la cabeza de valor de la red espera al terminar la mano">
        <div class="seccion-titulo">Valor estimado <span class="valor-num ${tone}">${signed(v, 1)} pts</span></div>
        <div class="gauge" aria-hidden="true"><div class="riel"></div><div class="cero"></div><div class="aguja" style="left:${pos}%"></div></div>
        <div class="gauge-escala"><span>−4</span><span>0</span><span>+4</span></div>
      </div>`);
    }
    if (thought.options && thought.kind !== 'rules') {
      const options = thought.options.slice(0, 8);
      const title = thought.kind === 'random' ? 'Al azar' : thought.kind === 'table' ? 'Estrategia MCCFR' : thought.kind === 'mute' ? 'Seña fija' : 'Política';
      const chosenLabel = advice ? 'sugerida' : 'elegida';
      const rest = thought.options.length - options.length;
      const rows = options.map((o) => {
        const chosen = o.action === thought.chosen;
        const width = clamp(o.p * 100, 0, 100);
        return `<li class="prob tipo-${actionKind(o.action)}${chosen ? ' elegida' : ''}">
          <span class="etq" title="${esc(o.label)}">${chosen ? `<span class="check" title="${chosenLabel}">✓</span>` : ''}${esc(o.label)}</span>
          <span class="barra"><i style="width:${width}%"></i></span>
          <span class="pct">${pct(o.p)}</span>
        </li>`;
      });
      const more = rest > 0 ? ` · ${rest} más` : '';
      parts.push(`<div><div class="seccion-titulo">${esc(title)} <span>${thought.options.length} legales${more}</span></div><ul class="probs">${rows.join('')}</ul></div>`);
    } else if (thought.kind === 'rules') {
      parts.push(`<div class="reglas-nota">Decide con reglas fijas, sin probabilidades. Eligió <b>${esc(thought.chosen_label)}</b>.</div>`);
    }
    if (thought.beliefs) {
      const rows = thought.beliefs.map((b) => {
        const width = clamp(b.p * 100, 0, 100);
        const base = clamp(b.base * 100, 0, 100);
        const high = b.p >= 0.5 && b.p > 2 * b.base;
        return `<span class="rasgo${high ? ' alto' : ''}" title="${esc(b.label)}">${esc(b.label)}</span>
          <span class="barra-base" title="Sin información: ${pct(b.base)}"><i style="width:${width}%"></i><b style="left:${base}%"></b></span>
          <span class="pct${high ? ' alto' : ''}">${pct(b.p)}</span>`;
      });
      const whose = advice ? 'tu compañero' : 'su compañero';
      parts.push(`<div title="Predicción de la cabeza auxiliar de la red. La rayita marca la probabilidad sin información.">
        <div class="seccion-titulo">Cree que ${whose} tiene</div>
        <div class="creencias">${rows.join('')}</div></div>`);
    }
    return `<div class="mente-card${advice ? ' consejo' : ''}">${parts.join('')}</div>`;
  }

  timeline() {
    if (!this.thoughts.length) return '';
    const start = Math.max(0, this.thoughts.length - 14);
    const chips = this.thoughts.slice(start).map((t, k) => {
      const index = start + k;
      const active = this.selected === index || (this.selected == null && !this.advice && index === this.thoughts.length - 1);
      return `<button type="button" data-pick="${index}" aria-pressed="${active}" title="${esc(this.game.seats[t.seat].role)}: ${esc(t.chosen_label)}">${miniFace(t.seat)}<span>${esc(this.short(t.chosen_label))}</span></button>`;
    });
    const follow = this.selected != null ? '<button type="button" class="seguir" data-pick="seguir">Seguir la última</button>' : '';
    return `<div class="linea-tiempo">${chips.join('')}${follow}</div>`;
  }

  short(label) {
    const text = label.split(' · ')[0];
    return text.length > 16 ? `${text.slice(0, 15)}…` : text;
  }
}
