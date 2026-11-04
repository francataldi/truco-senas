import { esc } from './dom.js';

const SIZE = 26;
const COL = 35;
const ROW = 40;
const X0 = 6;
const Y0 = 5;

function jitter(n) {
  const x = Math.sin(n * 12.9898) * 43758.5453;
  return (x - Math.floor(x) - 0.5) * 1.2;
}

function stick(index, x1, y1, x2, y2, extra, delay) {
  const j = jitter(index);
  const a = [x1 + j * 0.5, y1 + j * 0.3];
  const b = [x2 - j * 0.3, y2 + j * 0.5];
  const style = delay != null ? ` style="animation-delay:${delay}ms"` : '';
  return `<g class="palito${extra}"${style}>
    <line class="sombra" x1="${a[0] + 0.8}" y1="${a[1] + 1.2}" x2="${b[0] + 0.8}" y2="${b[1] + 1.2}"/>
    <line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}"/>
    <circle cx="${a[0]}" cy="${a[1]}" r="2.7"/>
  </g>`;
}

export function matchesSVG(points, target, previous = points) {
  const squares = Math.ceil(target / 5);
  const perRow = 3;
  const rows = Math.ceil(squares / perRow);
  const width = X0 * 2 + (perRow - 1) * COL + SIZE;
  const height = Y0 * 2 + (rows - 1) * ROW + SIZE;
  const parts = [];
  for (let s = 0; s < squares; s += 1) {
    const x = X0 + (s % perRow) * COL;
    const y = Y0 + Math.floor(s / perRow) * ROW;
    const filled = Math.max(0, Math.min(5, points - s * 5));
    if (filled === 0) {
      parts.push(`<rect class="vacio" x="${x}" y="${y}" width="${SIZE}" height="${SIZE}" rx="2"/>`);
      continue;
    }
    const L = x + 0.5;
    const R = x + SIZE - 0.5;
    const T = y + 0.5;
    const B = y + SIZE - 0.5;
    const segments = [
      [L, T + 2, L, B - 1],
      [L + 2, T, R - 1, T],
      [R, T + 2, R, B - 1],
      [L + 2, B, R - 1, B],
      [L + 3, T + 3, R - 3, B - 3],
    ];
    for (let k = 0; k < filled; k += 1) {
      const n = s * 5 + k;
      const fresh = n >= previous;
      const [x1, y1, x2, y2] = segments[k];
      parts.push(stick(n + 1, x1, y1, x2, y2, fresh ? ' nuevo' : '', fresh ? (n - previous) * 110 : null));
    }
  }
  for (let r = 1; r < rows; r += 1) {
    const y = Y0 + r * ROW - (ROW - SIZE) / 2;
    parts.push(`<line class="divisor" x1="2" y1="${y}" x2="${width - 2}" y2="${y}"/>`);
  }
  return `<svg class="fosforos" viewBox="0 0 ${width} ${height}" role="img" aria-label="${points} puntos">${parts.join('')}</svg>`;
}

export class Scoreboard {
  constructor(root, hint) {
    this.root = root;
    this.hint = hint;
    this.previous = [0, 0];
    this.target = 30;
    this.teams = [];
  }

  setup(game) {
    this.teams = game.teams;
    this.target = game.target;
    this.previous = [0, 0];
    if (this.hint) this.hint.textContent = this.target === 30 ? 'a 30, malas y buenas' : `a ${this.target}`;
    this.render([0, 0]);
  }

  render(scores) {
    if (!this.teams.length) return;
    const leader = scores[0] === scores[1] ? -1 : scores[0] > scores[1] ? 0 : 1;
    const cols = [0, 1].map((t) => {
      const team = this.teams[t];
      const points = Math.min(scores[t], this.target);
      return `<div class="equipo team-${t}${leader === t ? ' lider' : ''}">
        <div class="equipo-head"><span class="nombre">${esc(team.name)}</span><span class="puntos">${scores[t]}</span></div>
        <div class="equipo-agentes" title="${esc(team.agents)}">${esc(team.agents)}</div>
        ${matchesSVG(points, this.target, Math.min(this.previous[t], points))}
      </div>`;
    });
    const half = this.target / 2;
    const status = [0, 1].map((t) => {
      const name = this.teams[t].name;
      if (this.target !== 30) return `${name}: faltan ${Math.max(0, this.target - scores[t])}`;
      return `${name}: ${scores[t] >= half ? 'buenas' : 'malas'}`;
    });
    this.root.innerHTML = `<div class="equipos">${cols.join('')}</div>
      <div class="marcador-pie"><span>${esc(status[0])}</span><span>${esc(status[1])}</span></div>`;
    this.previous = [...scores];
  }
}
