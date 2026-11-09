import { api } from './api.js';
import { cardSVG, installSprite } from './cards.js';
import { Controls, SPEEDS } from './controls.js';
import { $, $$, esc, sleep, storage, store, toast } from './dom.js';
import { GESTURES } from './faces.js';
import { Log, MindPanel, SignalsPanel } from './panels.js';
import { Scoreboard } from './score.js';
import { Table } from './table.js';

const PAUSES = { deal: 650, signal: 1250, play: 380, call: 1350, answer: 1000, round: 800, envido: 1700, mazo: 1100 };
const FLIGHT = 540;
const SPY_OPTIONS = [
  [0, 'Nunca'],
  [0.1, '10%'],
  [0.2, '20%, como en el entrenamiento'],
  [0.35, '35%'],
  [0.5, '50%'],
];

const app = {
  agents: [],
  defaults: null,
  game: null,
  config: storage('truco-config', null),
  state: null,
  seq: 0,
  queue: [],
  pumpGen: null,
  generation: 0,
  speed: SPEEDS.includes(storage('truco-velocidad', 1)) ? storage('truco-velocidad', 1) : 1,
  paused: false,
  stepOnce: false,
  resume: null,
  humanTurn: false,
  turnId: 0,
  sending: false,
  handEnvido: null,
  setupMode: 'play',
  setupFormat: 'parejas',
};

const table = new Table($('#pano'));
const score = new Scoreboard($('#marcador'), $('#marcador-hint'));
const log = new Log($('#log'), $('#bitacora-hint'));
const signals = new SignalsPanel($('#senas'), $('#senas-hint'));
const mind = new MindPanel($('#mente'), $('#mente-toggle'));
const controls = new Controls($('#controles'));

const wait = (ms) => sleep(ms / app.speed);

function capital(text) {
  return text ? text[0].toUpperCase() + text.slice(1) : '';
}

function seatOf(index) {
  return app.game.seats[index];
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try {
    localStorage.setItem('truco-tema', theme);
  } catch (error) {
    return;
  }
}

function setModePill(text, color) {
  $('#mode-text').textContent = text;
  const dot = $('#mode-pill .dot');
  if (dot) dot.style.background = color || '';
}

function heroCards() {
  const box = $('#hero-cards');
  if (!box) return;
  box.innerHTML = [
    { rank: 7, suit: 'oro', name: '7 de oro' },
    { rank: 1, suit: 'espada', name: 'ancho de espada' },
    { rank: 1, suit: 'basto', name: 'ancho de basto' },
  ].map((card) => cardSVG(card)).join('');
}

function agentLabel(spec) {
  const info = app.agents.find((a) => a.spec === spec);
  return info ? info.label : spec;
}

function playsWith(agent, players) {
  return (agent.players || [4]).includes(players);
}

function optionsMarkup(selected, players = 4) {
  const fits = app.agents.filter((a) => playsWith(a, players));
  const groups = [
    ['Inteligencias artificiales', fits.filter((a) => a.family === 'ppo')],
    ['Teoría de juegos', fits.filter((a) => a.family === 'cfr')],
    ['Programados a mano', fits.filter((a) => a.family === 'heuristic' || a.family === 'random')],
  ];
  return groups
    .filter(([, list]) => list.length)
    .map(([label, list]) => `<optgroup label="${esc(label)}">${list
      .map((a) => `<option value="${esc(a.spec)}"${a.spec === selected ? ' selected' : ''}>${esc(a.label)}</option>`)
      .join('')}</optgroup>`)
    .join('');
}

function field(id, label, team, selected, wide = false, players = 4) {
  return `<div class="field${wide ? ' wide' : ''}">
    <label for="${id}"><span class="seat-dot" style="background:var(--team-${team})"></span>${esc(label)}</label>
    <select id="${id}">${optionsMarkup(selected, players)}</select>
    <div class="desc" data-desc="${id}"></div>
  </div>`;
}

const TWINS = { 'sel-left': ['sel-right', 'el rival derecho'], 'sel-b': ['sel-a', 'la pareja A'], 'sel-db': ['sel-da', 'el de Sur'] };

function describe(select) {
  const box = $(`[data-desc="${select.id}"]`);
  if (!box) return;
  const info = app.agents.find((a) => a.spec === select.value);
  if (!info) {
    box.textContent = '';
    return;
  }
  const head = info.summary ? `<b>${esc(info.summary)}</b> ` : '';
  const twin = TWINS[select.id];
  const same = twin && $(`#${twin[0]}`) && $(`#${twin[0]}`).value === select.value;
  box.innerHTML = head + esc(same ? `Es el mismo tipo de jugador que ${twin[1]}.` : info.description);
}

function savedConfig() {
  const fallback = { play: ['human', 'heuristic', 'heuristic', 'heuristic'], watch: ['heuristic', 'heuristic', 'heuristic', 'heuristic'], duel: ['human', 'heuristic'], duel_watch: ['heuristic', 'heuristic'] };
  const defaults = { ...fallback, ...(app.defaults || {}) };
  const base = {
    mode: 'play',
    format: 'parejas',
    rival: defaults.duel[1],
    da: defaults.duel_watch[0],
    db: defaults.duel_watch[1],
    partner: defaults.play[2],
    right: defaults.play[1],
    left: defaults.play[3],
    a: defaults.watch[0],
    b: defaults.watch[1],
    target: 30,
    open_cards: false,
    spy_prob: 0.2,
    scramble: 'none',
  };
  const saved = app.config || {};
  const merged = { ...base, ...saved };
  const fits = (spec, players) => app.agents.some((a) => a.spec === spec && playsWith(a, players));
  for (const key of ['partner', 'right', 'left', 'a', 'b']) {
    if (!fits(merged[key], 4)) merged[key] = base[key];
  }
  for (const key of ['rival', 'da', 'db']) {
    if (!fits(merged[key], 2)) merged[key] = base[key];
  }
  return merged;
}

function renderSetup() {
  const cfg = savedConfig();
  const mode = app.setupMode;
  const duel = app.setupFormat === 'duelo';
  $$('.segmented [data-mode]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.mode === mode)));
  $$('.segmented [data-format]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.format === app.setupFormat)));
  const help = {
    'play-parejas': 'Vos y una IA de compañera contra dos rivales.',
    'watch-parejas': 'Mirá jugar solas a dos parejas, a la velocidad que quieras.',
    'play-duelo': 'Vos solo contra el bot que elijas, sin compañeros ni señas.',
    'watch-duelo': 'Mirá a dos bots jugar mano a mano.',
  };
  $('#mode-help').textContent = help[`${mode}-${app.setupFormat}`];
  $('#setup-ok').textContent = mode === 'play' ? 'Repartir' : 'Empezar';
  const grid = $('#setup-grid');
  if (duel && mode === 'play') {
    grid.innerHTML = field('sel-rival', 'Tu rival', 1, cfg.rival, true, 2);
  } else if (duel) {
    grid.innerHTML = field('sel-da', 'Sur', 0, cfg.da, false, 2) + field('sel-db', 'Norte', 1, cfg.db, false, 2);
  } else if (mode === 'play') {
    grid.innerHTML = field('sel-partner', 'Tu compañero', 0, cfg.partner, true)
      + field('sel-right', 'Rival derecho', 1, cfg.right)
      + field('sel-left', 'Rival izquierdo', 1, cfg.left);
  } else {
    grid.innerHTML = field('sel-a', 'Pareja A · Sur y Norte', 0, cfg.a) + field('sel-b', 'Pareja B · Este y Oeste', 1, cfg.b);
  }
  $$('#setup-grid select').forEach((select) => {
    describe(select);
    select.addEventListener('change', () => $$('#setup-grid select').forEach(describe));
  });
  const spy = SPY_OPTIONS.map(([v, label]) => `<option value="${v}"${Math.abs(v - cfg.spy_prob) < 1e-9 ? ' selected' : ''}>${esc(label)}</option>`).join('');
  const targets = [15, 30].map((t) => `<button type="button" data-target="${t}" aria-pressed="${cfg.target === t}">A ${t}</button>`).join('');
  const extra = mode === 'play'
    ? `<label class="switch" title="Ves las cartas y la mente de ${duel ? 'tu rival' : 'los rivales'}"><input type="checkbox" id="opt-open"${cfg.open_cards ? ' checked' : ''}><span class="track"></span>Cartas abiertas</label>`
    : duel ? '' : `<div class="field"><label for="opt-scramble">Experimento</label><select id="opt-scramble">
        <option value="none"${cfg.scramble === 'none' ? ' selected' : ''}>Señas normales</option>
        <option value="0"${cfg.scramble === '0' ? ' selected' : ''}>Mezclar las señas de la pareja A</option>
        <option value="1"${cfg.scramble === '1' ? ' selected' : ''}>Mezclar las señas de la pareja B</option>
      </select></div>`;
  $('#setup-options').innerHTML = `
    <div class="field"><span class="label">Puntos</span><div class="segmented" role="group" aria-label="Puntos">${targets}</div></div>
    ${duel ? '' : `<div class="field"><label for="opt-spy" title="Chance de que el rival vea la seña que le hacés a tu compañero, como en una mesa de verdad">Chance de que el rival vea tu seña</label><select id="opt-spy">${spy}</select></div>`}
    ${extra}`;
  $$('#setup-options [data-target]').forEach((b) => b.addEventListener('click', () => {
    $$('#setup-options [data-target]').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
  }));
  const players = duel ? 2 : 4;
  const ready = app.agents.filter((a) => (a.kind === 'ppo' || a.kind === 'cfr') && playsWith(a, players)).length;
  $('#setup-note').textContent = ready
    ? `${ready} ${ready === 1 ? 'bot entrenado listo' : 'bots entrenados listos'} para ${duel ? 'el mano a mano' : 'jugar en parejas'}`
    : duel ? 'Para el mano a mano no hay bots entrenados: podés jugar contra el clásico o el jugador al azar.' : 'No encontré redes entrenadas en runs/: entrená una o pasá --checkpoint.';
}

function readSetup() {
  const cfg = savedConfig();
  cfg.mode = app.setupMode;
  cfg.format = app.setupFormat;
  if (cfg.format === 'duelo') {
    if (cfg.mode === 'play') {
      cfg.rival = $('#sel-rival').value;
      cfg.open_cards = $('#opt-open').checked;
    } else {
      cfg.da = $('#sel-da').value;
      cfg.db = $('#sel-db').value;
    }
  } else if (cfg.mode === 'play') {
    cfg.partner = $('#sel-partner').value;
    cfg.right = $('#sel-right').value;
    cfg.left = $('#sel-left').value;
    cfg.open_cards = $('#opt-open').checked;
  } else {
    cfg.a = $('#sel-a').value;
    cfg.b = $('#sel-b').value;
    cfg.scramble = $('#opt-scramble').value;
  }
  const target = $('#setup-options [data-target][aria-pressed="true"]');
  cfg.target = target ? Number(target.dataset.target) : 30;
  if ($('#opt-spy')) cfg.spy_prob = Number($('#opt-spy').value);
  return cfg;
}

function openSetup(mode = null) {
  const dialog = $('#setup');
  app.setupMode = mode || (app.config && app.config.mode) || 'play';
  app.setupFormat = (app.game && app.game.players === 2) || (!app.game && app.config && app.config.format === 'duelo') ? 'duelo' : app.setupFormat;
  $('#setup-error').hidden = true;
  renderSetup();
  if (!dialog.open) dialog.showModal();
}

function bodyFor(cfg) {
  if (cfg.format === 'duelo') {
    return cfg.mode === 'play'
      ? { mode: 'play', seats: ['human', cfg.rival], target: cfg.target, open_cards: cfg.open_cards }
      : { mode: 'watch', seats: [cfg.da, cfg.db], target: cfg.target };
  }
  if (cfg.mode === 'play') {
    return {
      mode: 'play',
      seats: ['human', cfg.right, cfg.partner, cfg.left],
      target: cfg.target,
      open_cards: cfg.open_cards,
      spy_prob: cfg.spy_prob,
    };
  }
  return {
    mode: 'watch',
    seats: [cfg.a, cfg.b, cfg.a, cfg.b],
    target: cfg.target,
    spy_prob: cfg.spy_prob,
    scramble: cfg.scramble === 'none' ? [] : [Number(cfg.scramble)],
  };
}

async function startGame(cfg) {
  const gen = ++app.generation;
  table.resolveSummary('abort');
  if (app.resume) app.resume();
  app.queue = [];
  app.paused = false;
  app.stepOnce = false;
  app.humanTurn = false;
  const ok = $('#setup-ok');
  ok.disabled = true;
  ok.textContent = 'Preparando…';
  let data;
  try {
    data = await api.newGame(bodyFor(cfg));
  } catch (error) {
    ok.disabled = false;
    renderSetup();
    const box = $('#setup-error');
    box.textContent = error.message;
    box.hidden = false;
    if (!$('#setup').open) {
      openSetup(cfg.mode);
      $('#setup-error').textContent = error.message;
      $('#setup-error').hidden = false;
    }
    return false;
  }
  ok.disabled = false;
  if (gen !== app.generation) return false;
  const previous = app.game;
  app.config = cfg;
  store('truco-config', cfg);
  app.game = data;
  app.seq = data.seq;
  app.state = null;
  if (previous) api.remove(previous.id).catch(() => null);
  $('#setup').close();
  setupUI(data);
  enqueue(data.events);
  pump(gen);
  return true;
}

function setupUI(game) {
  table.setup(game);
  table.showAll = true;
  table.updateToggle();
  table.onPlay = (action) => humanAction(action);
  score.setup(game);
  log.clear();
  signals.setup(game);
  mind.setup(game);
  controls.setup(game, app.speed);
  controls.setPaused(false);
  if (game.mode === 'play') setModePill(`Jugando a ${game.target}`, 'var(--good)');
  else setModePill(`Mirando a ${game.target}`, 'var(--envido)');
  document.title = game.mode === 'play' ? 'Truco con señas · jugando' : 'Truco con señas · mirando';
}

function enqueue(events) {
  for (const event of events) app.queue.push(event);
}

function waitResume() {
  return new Promise((resolve) => {
    app.resume = () => {
      app.resume = null;
      resolve();
    };
  });
}

async function pump(gen) {
  if (app.pumpGen === gen) return;
  app.pumpGen = gen;
  try {
    while (gen === app.generation) {
      if (app.queue.length) {
        if (app.game.mode === 'watch' && app.paused && !app.stepOnce) {
          await waitResume();
          continue;
        }
        app.stepOnce = false;
        await handle(app.queue.shift(), gen);
        continue;
      }
      const state = app.state;
      if (!state || state.phase === 'match_over') break;
      if (state.phase === 'hand_over') {
        if (app.game.mode === 'watch' && app.paused && !app.stepOnce) {
          await waitResume();
          continue;
        }
        if (!(await advance(gen))) break;
        continue;
      }
      if (app.game.mode === 'play') {
        if (state.turn === app.game.viewer) beginHumanTurn(state, gen);
        break;
      }
      if (app.paused && !app.stepOnce) {
        await waitResume();
        continue;
      }
      if (!(await advance(gen))) break;
    }
  } catch (error) {
    if (gen === app.generation) lost(error);
  } finally {
    if (app.pumpGen === gen) app.pumpGen = null;
  }
}

function lost(error) {
  if (error.status === 404) {
    toast('Esta partida ya no está en el servidor. Empezá una nueva.', 'error');
    app.generation += 1;
    openSetup(app.game ? app.game.mode : null);
    return;
  }
  toast(error.message || 'Algo salió mal', 'error');
}

async function advance(gen) {
  if (app.game.mode === 'watch') controls.status(`Mano ${app.state ? app.state.hand : ''}`, 'Los agentes están pensando', true);
  let data;
  try {
    data = await api.advance(app.game.id, 64, app.seq);
  } catch (error) {
    if (error.status === 409) return resync(gen);
    throw error;
  }
  if (gen !== app.generation) return false;
  app.seq = data.seq;
  enqueue(data.events);
  return data.events.length > 0;
}

async function resync(gen) {
  const data = await api.game(app.game.id, app.seq);
  if (gen !== app.generation) return false;
  app.seq = data.seq;
  enqueue(data.events);
  return data.events.length > 0;
}

function bubbleForSignal(event) {
  if (event.token == null) return ['hace una seña…', 'sena oculta'];
  const language = seatOf(event.seat).language;
  const entry = language && language.tokens[event.token];
  const gesture = capital(GESTURES[event.token] || `seña ${event.token}`);
  const meaning = entry && entry.confidence != null && language.source !== 'mudo' ? ` · ${entry.label}` : '';
  return [`${gesture}${meaning}`, 'sena'];
}

async function handle(event, gen) {
  if (gen !== app.generation) return;
  switch (event.type) {
    case 'deal':
      table.hideOverlay();
      signals.newHand();
      mind.newHand();
      app.handEnvido = null;
      table.render(event.state, { deal: true });
      break;
    case 'play': {
      const from = table.sourceRect(event.seat, event.slot);
      table.render(event.state);
      await table.flyCard(event.seat, event.round, from, FLIGHT / app.speed);
      if (gen !== app.generation) return;
      break;
    }
    case 'signal': {
      table.render(event.state);
      if (event.token != null) table.gesture(event.seat, event.token, 1800 / app.speed);
      const [text, kind] = bubbleForSignal(event);
      table.bubble(event.seat, text, kind, 1900 / app.speed);
      break;
    }
    case 'call':
      table.render(event.state);
      table.bubble(event.seat, event.say, event.call === 'truco' ? 'truco' : 'envido', 2300 / app.speed);
      break;
    case 'answer':
      table.render(event.state);
      table.bubble(event.seat, event.say, event.accepted ? 'quiero' : 'no-quiero', 1900 / app.speed);
      break;
    case 'mazo':
      table.render(event.state);
      table.bubble(event.seat, event.say, 'mazo', 2000 / app.speed);
      break;
    case 'round':
      table.render(event.state);
      if (event.winner != null) table.pulseWinner(event.seat, event.round);
      break;
    case 'envido':
      app.handEnvido = event;
      table.render(event.state);
      if (event.accepted && event.seat != null) table.bubble(event.seat, `Tengo ${event.shown[String(event.seat)]}`, 'envido', 2100 / app.speed);
      break;
    default:
      table.render(event.state);
  }
  if (gen !== app.generation) return;
  app.state = event.state;
  score.render(event.state.scores);
  log.add(event, app.game);
  signals.update(event.state, event);
  mind.push(event);
  statusFor(event);
  if (event.type === 'hand_end') await handEnd(event, gen);
  else if (event.type === 'match_end') await matchEnd(event, gen);
  else await wait(PAUSES[event.type] ?? 600);
}

function statusFor(event) {
  const state = event.state;
  if (app.game.mode === 'watch') {
    controls.status(`Mano ${state.hand}`, event.text);
    return;
  }
  if (state.phase === 'play' || state.phase === 'signals') {
    if (state.turn !== app.game.viewer) {
      const who = seatOf(state.turn).role;
      let title = `Juega ${who}`;
      if (state.phase === 'signals') title = `${who} hace su seña`;
      else if (state.pending) title = `${who} responde`;
      controls.status(title, event.text, true);
      controls.clear();
    }
  }
}

async function handEnd(event, gen) {
  table.setLocked(true);
  if (event.state.phase === 'match_over') {
    await wait(900);
    return;
  }
  const team = event.winner;
  const name = app.game.teams[team].name;
  const play = app.game.mode === 'play';
  const title = play ? (team === 0 ? '¡Mano para nosotros!' : 'Mano para ellos') : `Mano para la ${name}`;
  controls.status('Fin de la mano', play ? 'Mirá las cartas de todos y seguí cuando quieras.' : event.text);
  controls.clear();
  const envido = app.handEnvido;
  const extra = envido ? ` · Envido${envido.accepted ? '' : ' no querido'}: +${envido.value} para ${app.game.teams[envido.winner].name}` : '';
  const choice = await table.summary({
    eyebrow: `Mano ${event.state.hand} · ${title.replace(/[¡!]/g, '')}`,
    title: play ? (team === 0 ? `+${event.points[0]} para nosotros` : `+${event.points[1]} para ellos`) : `+${event.points[team]} para la ${name}`,
    detail: `${event.text}${extra}`,
    points: event.points,
    buttons: [{ id: 'next', label: play ? 'Siguiente mano' : 'Seguir', primary: true, key: 'Enter' }],
    auto: play || app.paused ? 0 : 3400 / app.speed,
    docked: true,
  });
  if (gen !== app.generation || choice === 'abort') return;
  table.hideOverlay();
  controls.clear();
}

async function matchEnd(event, gen) {
  const play = app.game.mode === 'play';
  const won = event.winner === 0;
  const scores = event.scores;
  const teams = app.game.teams;
  setModePill('Partida terminada', 'var(--gold)');
  controls.status(play ? (won ? '¡Ganaron!' : 'Perdieron') : 'Fin de la partida', `${teams[0].name} ${scores[0]} · ${teams[1].name} ${scores[1]}`);
  controls.buttons([
    { ctl: 'rematch', label: 'Revancha' },
    { ctl: 'setup', label: 'Cambiar agentes', cls: 'mazo' },
  ]);
  const choice = await table.summary({
    final: true,
    eyebrow: 'Fin de la partida',
    title: event.text,
    detail: `${teams[0].name} ${scores[0]} · ${teams[1].name} ${scores[1]} en ${event.state.hand} manos`,
    buttons: [
      { id: 'rematch', label: 'Revancha', primary: true },
      { id: 'setup', label: 'Cambiar agentes' },
      { id: 'view', label: 'Ver la mesa' },
    ],
  });
  if (gen !== app.generation || choice === 'abort') return;
  if (choice === 'rematch') startGame(app.config);
  else if (choice === 'setup') openSetup(app.game.mode);
  else table.hideOverlay();
}

async function beginHumanTurn(state, gen) {
  app.humanTurn = true;
  const turn = ++app.turnId;
  const current = () => gen === app.generation && app.humanTurn && turn === app.turnId;
  table.setLocked(false);
  const partner = seatOf(2);
  if (state.phase === 'signals') {
    controls.status('Tu seña', partner.mute ? 'Tu compañero no usa señas: cualquier gesto da igual.' : 'Cada gesto muestra lo que suele anunciar para tu compañero: es una pista, no una garantía.');
    controls.picker(state, null, partner.mute);
    try {
      const { advice } = await api.advice(app.game.id);
      if (!current() || !advice) return;
      if (mind.enabled) mind.setAdvice(advice);
      if (!partner.mute && advice.chosen >= 11) {
        const option = advice.kind === 'policy' ? advice.options.find((o) => o.action === advice.chosen) : null;
        controls.picker(state, { token: advice.chosen - 11, p: option ? option.p : null }, partner.mute);
      }
    } catch (error) {
      return;
    }
    return;
  }
  const pending = state.pending;
  const mine = state.seats[app.game.viewer].envido;
  if (pending && pending.kind === 'truco') controls.status(`Te cantaron ${pending.name}`, '¿Querés? También podés subir la apuesta.');
  else if (pending) controls.status(`Te cantaron ${pending.name}`, `Tenés ${mine} de envido.`);
  else controls.status('Tu turno', state.legal.some((o) => o.kind === 'card') ? 'Jugá una carta o cantá.' : 'Elegí qué hacer.');
  controls.actions(state);
  if (mind.enabled) {
    try {
      const { advice } = await api.advice(app.game.id);
      if (current() && advice) mind.setAdvice(advice);
    } catch (error) {
      return;
    }
  }
}

async function humanAction(action) {
  if (!app.humanTurn || app.sending || !app.game) return;
  const state = app.state;
  if (!state || !state.legal.some((o) => o.action === action)) return;
  app.sending = true;
  app.humanTurn = false;
  table.setLocked(true);
  controls.clear();
  controls.status('Jugando…', '', true);
  mind.clearAdvice();
  const gen = app.generation;
  try {
    const data = await api.act(app.game.id, action, app.seq);
    if (gen !== app.generation) return;
    app.seq = data.seq;
    enqueue(data.events);
  } catch (error) {
    if (error.status === 404) {
      lost(error);
      return;
    }
    toast(error.message, 'error');
    if (error.status === 409) await resync(gen).catch(() => null);
    else {
      app.humanTurn = false;
      beginHumanTurn(state, gen);
    }
  } finally {
    app.sending = false;
  }
  pump(gen);
}

function playKey(index) {
  const state = app.state;
  const own = state.seats[app.game.viewer];
  const visible = own.cards.map((card, slot) => ({ card, slot })).filter(({ slot }) => !own.played[slot]);
  const pick = visible[index];
  if (!pick) return;
  const option = state.legal.find((o) => o.kind === 'card' && o.slot === pick.slot);
  if (option) humanAction(option.action);
}

function togglePause() {
  if (!app.game || app.game.mode !== 'watch') return;
  app.paused = !app.paused;
  controls.setPaused(app.paused);
  setModePill(app.paused ? 'En pausa' : `Mirando a ${app.game.target}`, app.paused ? 'var(--gold)' : 'var(--envido)');
  if (!app.paused && app.resume) app.resume();
  if (!app.paused) pump(app.generation);
}

function step() {
  if (!app.game || app.game.mode !== 'watch' || !app.paused) return;
  app.stepOnce = true;
  if (app.resume) app.resume();
  else pump(app.generation);
  if (table.pendingSummary) table.resolveSummary('next');
}

function onKey(event) {
  if ($('#setup').open || event.altKey || event.ctrlKey || event.metaKey) return;
  if (event.target.closest('input, select, textarea')) return;
  const key = event.key.toLowerCase();
  if (table.pendingSummary && (key === 'enter' || (key === ' ' && app.game.mode === 'play'))) {
    event.preventDefault();
    table.resolveSummary('next');
    return;
  }
  if (!app.game) return;
  if (app.game.mode === 'watch') {
    if (key === ' ') {
      event.preventDefault();
      togglePause();
    } else if (key === 'arrowright') {
      step();
    }
    return;
  }
  if (!app.humanTurn) return;
  const state = app.state;
  if (state.phase === 'signals') {
    const n = Number(key);
    if (n >= 1 && n <= 8) {
      const option = state.legal.filter((o) => o.kind === 'signal')[n - 1];
      if (option) humanAction(option.action);
    }
    return;
  }
  if (key === '1' || key === '2' || key === '3') {
    playKey(Number(key) - 1);
    return;
  }
  const map = { t: 3, e: 4, r: 5, f: 6, q: 7, n: 8, m: 9 };
  if (map[key] != null) humanAction(map[key]);
}

function bind() {
  $('#btn-tema').addEventListener('click', () => setTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'));
  $('#btn-nueva').addEventListener('click', () => openSetup(app.game ? app.game.mode : null));
  $$('.segmented [data-mode]').forEach((b) => b.addEventListener('click', () => {
    app.setupMode = b.dataset.mode;
    $('#setup-error').hidden = true;
    renderSetup();
  }));
  $$('.segmented [data-format]').forEach((b) => b.addEventListener('click', () => {
    app.setupFormat = b.dataset.format;
    $('#setup-error').hidden = true;
    renderSetup();
  }));
  $('#setup-form').addEventListener('submit', (event) => {
    event.preventDefault();
    startGame(readSetup());
  });
  $('#setup-cancelar').addEventListener('click', () => $('#setup').close());
  $('#setup').addEventListener('cancel', (event) => {
    if (!app.game) event.preventDefault();
  });
  $$('.tabs [data-tab]').forEach((b) => b.addEventListener('click', () => {
    $('#layout').dataset.tab = b.dataset.tab;
    $$('.tabs [data-tab]').forEach((x) => x.setAttribute('aria-selected', String(x === b)));
  }));
  controls.on('action', (action) => humanAction(action));
  controls.on('next', () => table.resolveSummary('next'));
  controls.on('rematch', () => (table.pendingSummary ? table.resolveSummary('rematch') : startGame(app.config)));
  controls.on('setup', () => (table.pendingSummary ? table.resolveSummary('setup') : openSetup(app.game.mode)));
  controls.on('pause', togglePause);
  controls.on('step', step);
  controls.on('speed', (speed) => {
    app.speed = speed;
    store('truco-velocidad', speed);
  });
  mind.onToggle = (enabled) => {
    if (enabled && app.humanTurn && app.state) beginHumanTurn(app.state, app.generation);
  };
  document.addEventListener('keydown', onKey);
}

async function init() {
  installSprite();
  heroCards();
  bind();
  mind.render();
  try {
    const data = await api.agents();
    app.agents = data.agents;
    app.defaults = data.defaults;
  } catch (error) {
    $('#pano').innerHTML = `<div class="vacio-mesa">${esc(error.message)}</div>`;
    return;
  }
  $('#pano').innerHTML = '<div class="vacio-mesa">Elegí con quién jugar para repartir.</div>';
  openSetup();
}

init();
