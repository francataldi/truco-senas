const NS = 'http://www.w3.org/2000/svg';

const SUITS = {
  oro: { ink: '#8a5d05', gaps: 0, robe: '#b3342c', robeDark: '#76201b' },
  copa: { ink: '#8d241b', gaps: 1, robe: '#2f5f99', robeDark: '#1d3d66' },
  espada: { ink: '#1f4470', gaps: 2, robe: '#2e7553', robeDark: '#1b4a33' },
  basto: { ink: '#5a3816', gaps: 3, robe: '#c0612b', robeDark: '#7d3b15' },
};

const PIPS = {
  1: [[50, 75]],
  2: [[50, 43], [50, 107]],
  3: [[50, 36], [50, 75], [50, 114]],
  4: [[33, 43], [67, 43], [33, 107], [67, 107]],
  5: [[33, 40], [67, 40], [50, 75], [33, 110], [67, 110]],
  6: [[33, 37], [67, 37], [33, 75], [67, 75], [33, 113], [67, 113]],
  7: [[33, 34], [67, 34], [50, 55], [33, 76], [67, 76], [33, 114], [67, 114]],
};

const SPRITE = `
<svg xmlns="${NS}" id="card-sprite" aria-hidden="true" focusable="false" width="0" height="0" style="position:absolute;width:0;height:0;overflow:hidden">
  <defs>
    <linearGradient id="g-paper" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#fffdf6"/>
      <stop offset="1" stop-color="#f3ead6"/>
    </linearGradient>
    <radialGradient id="g-oro" cx="0.38" cy="0.32" r="0.75">
      <stop offset="0" stop-color="#ffeaa0"/>
      <stop offset="0.55" stop-color="#f0bd36"/>
      <stop offset="1" stop-color="#c48c0d"/>
    </radialGradient>
    <linearGradient id="g-copa" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#e2574a"/>
      <stop offset="0.6" stop-color="#c2362b"/>
      <stop offset="1" stop-color="#8f2219"/>
    </linearGradient>
    <linearGradient id="g-blade" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#cfe0f3"/>
      <stop offset="0.5" stop-color="#6f9bd0"/>
      <stop offset="1" stop-color="#2f5c92"/>
    </linearGradient>
    <linearGradient id="g-wood" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#d39a5c"/>
      <stop offset="0.55" stop-color="#a5693a"/>
      <stop offset="1" stop-color="#6e4220"/>
    </linearGradient>
    <linearGradient id="g-gold" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#ffe08a"/>
      <stop offset="1" stop-color="#d49b1c"/>
    </linearGradient>
    <radialGradient id="g-glow" cx="0.5" cy="0.5" r="0.5">
      <stop offset="0" stop-color="#ffd76a" stop-opacity="0.55"/>
      <stop offset="0.7" stop-color="#ffd76a" stop-opacity="0.12"/>
      <stop offset="1" stop-color="#ffd76a" stop-opacity="0"/>
    </radialGradient>
    <pattern id="p-back" width="12" height="12" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
      <rect width="12" height="12" fill="#7a1f24"/>
      <path d="M0 0H12M0 0V12" stroke="#c99a3b" stroke-width="0.9" opacity="0.55"/>
      <circle cx="6" cy="6" r="1.3" fill="#e6bf62" opacity="0.7"/>
    </pattern>
    <symbol id="s-oro" viewBox="-20 -20 40 40">
      <circle r="17.6" fill="url(#g-oro)" stroke="#7d5405" stroke-width="1.5"/>
      <circle r="13.4" fill="none" stroke="#a87507" stroke-width="1.1"/>
      <circle r="15.5" fill="none" stroke="#fff1bd" stroke-width="0.9" stroke-dasharray="0.1 3" stroke-linecap="round"/>
      <path d="M0-9.5 2.3-2.3 9.5 0 2.3 2.3 0 9.5-2.3 2.3-9.5 0-2.3-2.3Z" fill="#c4900f" stroke="#8a5d05" stroke-width="0.7" stroke-linejoin="round"/>
      <path d="M0-6.2 1.2-1.2 6.2 0 1.2 1.2 0 6.2-1.2 1.2-6.2 0-1.2-1.2Z" fill="#ffe59a" transform="rotate(45)"/>
      <circle r="2.1" fill="#8a5d05"/>
      <path d="M-11.5-8.5A14 14 0 0 1 7-12.2" fill="none" stroke="#fff7d6" stroke-width="1.5" stroke-linecap="round" opacity="0.85"/>
    </symbol>
    <symbol id="s-copa" viewBox="-20 -20 40 40">
      <path d="M-13.2-14.6H13.2C13.2-3.2 7.4 2.6 2.6 4.4V9.2C2.6 11.2 6.2 12.9 9 16.4H-9C-6.2 12.9-2.6 11.2-2.6 9.2V4.4C-7.4 2.6-13.2-3.2-13.2-14.6Z" fill="url(#g-copa)" stroke="#6d1811" stroke-width="1.3" stroke-linejoin="round"/>
      <path d="M-9.4-10C-8.8-3.6-5.6-0.4-2.4 1.2" fill="none" stroke="#f6a597" stroke-width="1.6" stroke-linecap="round" opacity="0.9"/>
      <path d="M-13.2-9.6C-6-7.6 6-7.6 13.2-9.6" fill="none" stroke="#f2c24f" stroke-width="1.6"/>
      <ellipse cy="-14.6" rx="13.2" ry="2.5" fill="url(#g-gold)" stroke="#7d5405" stroke-width="0.9"/>
      <rect x="-5.2" y="5.4" width="10.4" height="2.4" rx="1.2" fill="url(#g-gold)" stroke="#7d5405" stroke-width="0.6"/>
      <path d="M-9.6 16.2H9.6L10.4 18.4H-10.4Z" fill="url(#g-gold)" stroke="#7d5405" stroke-width="0.8" stroke-linejoin="round"/>
    </symbol>
    <symbol id="s-espada" viewBox="-20 -20 40 40">
      <path d="M0-19.6 3.1-14V7H-3.1V-14Z" fill="url(#g-blade)" stroke="#1b3a5f" stroke-width="1.1" stroke-linejoin="round"/>
      <path d="M0-14.5V5.6" stroke="#e8f1fb" stroke-width="0.9" opacity="0.8"/>
      <path d="M-10.5 7.2C-6.5 5.6 6.5 5.6 10.5 7.2L11.2 10C6.8 8.8-6.8 8.8-11.2 10Z" fill="url(#g-gold)" stroke="#7d5405" stroke-width="0.9" stroke-linejoin="round"/>
      <circle cx="-11.2" cy="8.6" r="1.6" fill="#f2c24f" stroke="#7d5405" stroke-width="0.6"/>
      <circle cx="11.2" cy="8.6" r="1.6" fill="#f2c24f" stroke="#7d5405" stroke-width="0.6"/>
      <rect x="-1.9" y="9.6" width="3.8" height="6.6" rx="1.2" fill="#5a3416" stroke="#331c09" stroke-width="0.6"/>
      <path d="M-1.9 11.6H1.9M-1.9 13.6H1.9" stroke="#8a5a2b" stroke-width="0.6"/>
      <circle cy="18" r="2.5" fill="url(#g-gold)" stroke="#7d5405" stroke-width="0.8"/>
    </symbol>
    <symbol id="s-basto" viewBox="-20 -20 40 40">
      <path d="M-2.7 19C-3.5 10-5.5-1-6.7-10C-7.5-16-4.3-19.6 0-19.6S7.5-16 6.7-10C5.5-1 3.5 10 2.7 19Z" fill="url(#g-wood)" stroke="#4a2c12" stroke-width="1.2" stroke-linejoin="round"/>
      <path d="M-1.6-16C-2.4-7-1.7 3-0.9 16" fill="none" stroke="#e2b27a" stroke-width="1" stroke-linecap="round" opacity="0.8"/>
      <ellipse cx="-6.1" cy="-6.5" rx="1.9" ry="1.3" fill="#6b4019" stroke="#4a2c12" stroke-width="0.5"/>
      <ellipse cx="5.4" cy="-1.5" rx="1.7" ry="1.2" fill="#6b4019" stroke="#4a2c12" stroke-width="0.5"/>
      <ellipse cx="-4.3" cy="5.8" rx="1.5" ry="1.1" fill="#6b4019" stroke="#4a2c12" stroke-width="0.5"/>
      <ellipse cx="3.5" cy="11" rx="1.3" ry="1" fill="#6b4019" stroke="#4a2c12" stroke-width="0.5"/>
      <path d="M6.4-12.6C9.4-16 13.2-16.4 15-14.4 12.6-11.8 9.2-11.2 6.4-12.6Z" fill="#5f8f3e" stroke="#35571f" stroke-width="0.6"/>
      <path d="M-6.6-14.4C-9.6-17.6-13.2-17.4-14.6-15.2-12-13-8.8-12.8-6.6-14.4Z" fill="#6f9f48" stroke="#35571f" stroke-width="0.6"/>
    </symbol>
    <symbol id="f-rey" viewBox="0 0 60 90">
      <path d="M17 40C11 52 8 70 7 88H53C52 70 49 52 43 40C38 37 22 37 17 40Z" fill="var(--robe)" stroke="var(--robe-dark)" stroke-width="1.4" stroke-linejoin="round"/>
      <path d="M30 39V88" stroke="#f2c24f" stroke-width="3"/>
      <path d="M7 86C20 82 40 82 53 86" fill="none" stroke="#f2c24f" stroke-width="2.6"/>
      <path d="M17 40C22 45 38 45 43 40" fill="none" stroke="#fff3d1" stroke-width="2.4"/>
      <circle cx="30" cy="27" r="8.6" fill="#f1c9a5" stroke="#9a6b48" stroke-width="1"/>
      <path d="M22.2 28C22 37 26 41 30 41S38 37 37.8 28C35 33 25 33 22.2 28Z" fill="#e9e4da" stroke="#9b958a" stroke-width="0.8"/>
      <circle cx="27" cy="26" r="0.9" fill="#3b2a1d"/>
      <circle cx="33" cy="26" r="0.9" fill="#3b2a1d"/>
      <path d="M21 19L22.5 9 26 15 30 7 34 15 37.5 9 39 19Z" fill="url(#g-gold)" stroke="#7d5405" stroke-width="1" stroke-linejoin="round"/>
      <circle cx="30" cy="16" r="1.4" fill="#c2362b"/>
      <circle cx="13" cy="58" r="3.2" fill="#f1c9a5" stroke="#9a6b48" stroke-width="0.8"/>
    </symbol>
    <symbol id="f-caballo" viewBox="0 0 60 90">
      <path d="M14 88C14 72 17 60 22 52 18 48 16 41 18 34 20 27 26 22 33 21L37 13 40 21C47 24 51 31 52 38L54 46C54.6 49 52 51 49.4 49.6L44 46C42 52 41 60 42 70 43 77 45 83 46 88Z" fill="#efe8db" stroke="#5b4a3a" stroke-width="1.4" stroke-linejoin="round"/>
      <path d="M33 21C27 26 24 34 25 44 26 52 23 60 20 70" fill="none" stroke="var(--robe)" stroke-width="5.5" stroke-linecap="round"/>
      <path d="M33 21C29 25 27 31 27.4 38" fill="none" stroke="var(--robe-dark)" stroke-width="2" stroke-linecap="round"/>
      <circle cx="43.4" cy="33.4" r="1.6" fill="#2e2219"/>
      <path d="M51 44.5C49.4 45.2 48 44.6 47.4 43.6" fill="none" stroke="#5b4a3a" stroke-width="1.1" stroke-linecap="round"/>
      <path d="M35 28L50 41M38.6 44.6 47 40" fill="none" stroke="#f2c24f" stroke-width="1.8" stroke-linecap="round"/>
      <path d="M14 78C24 74 36 74 46 78L46 88H14Z" fill="var(--robe)" stroke="var(--robe-dark)" stroke-width="1.2"/>
      <path d="M14 80C24 76 36 76 46 80" fill="none" stroke="#f2c24f" stroke-width="2"/>
    </symbol>
    <symbol id="f-sota" viewBox="0 0 60 90">
      <path d="M20 41C16 50 15 60 16 70H44C45 60 44 50 40 41C36 38 24 38 20 41Z" fill="var(--robe)" stroke="var(--robe-dark)" stroke-width="1.4" stroke-linejoin="round"/>
      <path d="M16 66H44" stroke="#f2c24f" stroke-width="2.4"/>
      <path d="M24 41 30 50 36 41" fill="none" stroke="#fff3d1" stroke-width="2.2" stroke-linejoin="round"/>
      <path d="M22 70 21 88M38 70 39 88" stroke="var(--robe-dark)" stroke-width="5" stroke-linecap="round"/>
      <path d="M17 88H26M34 88H43" stroke="#3b2a1d" stroke-width="3" stroke-linecap="round"/>
      <circle cx="30" cy="29" r="8.2" fill="#f1c9a5" stroke="#9a6b48" stroke-width="1"/>
      <path d="M22 29C21 23 25 19 30 19S39 23 38 29C36 25 33 24 30 24S24 25 22 29Z" fill="#7a4a24"/>
      <circle cx="27.2" cy="29" r="0.9" fill="#3b2a1d"/>
      <circle cx="32.8" cy="29" r="0.9" fill="#3b2a1d"/>
      <path d="M20 21C21 14 39 14 40 21 34 18 26 18 20 21Z" fill="var(--robe)" stroke="var(--robe-dark)" stroke-width="1"/>
      <path d="M38 18C44 10 50 9 52 11 49 14 44 17 38 19Z" fill="#f2c24f" stroke="#7d5405" stroke-width="0.8"/>
      <circle cx="46" cy="52" r="3" fill="#f1c9a5" stroke="#9a6b48" stroke-width="0.8"/>
    </symbol>
  </defs>
</svg>`;

let installed = false;

export function installSprite() {
  if (installed || document.getElementById('card-sprite')) {
    installed = true;
    return;
  }
  document.body.insertAdjacentHTML('afterbegin', SPRITE);
  installed = true;
}

function pip(suit, x, y, size, rotate = 0) {
  const half = size / 2;
  const turn = rotate ? ` rotate(${rotate})` : '';
  return `<use href="#s-${suit}" x="${-half}" y="${-half}" width="${size}" height="${size}" transform="translate(${x} ${y})${turn}"/>`;
}

function tilt(suit, index, x) {
  if (suit === 'espada') return x < 50 ? -16 : x > 50 ? 16 : 0;
  if (suit === 'basto') return x < 50 ? 14 : x > 50 ? -14 : 0;
  return 0;
}

const ACE_SIZE = { espada: 96, basto: 90, copa: 74, oro: 66 };
const PIP_SIZE = { espada: 31, basto: 31, copa: 28, oro: 27 };
const DUO_SIZE = { espada: 40, basto: 40, copa: 34, oro: 33 };

function aceArt(suit) {
  const glow = suit === 'espada' || suit === 'basto';
  const halo = glow ? '<circle cx="50" cy="75" r="44" fill="url(#g-glow)"/>' : '';
  const ring = suit === 'oro' ? '<circle cx="50" cy="75" r="38" fill="none" stroke="#d8b25a" stroke-width="0.8" stroke-dasharray="1 3" stroke-linecap="round"/>' : '';
  return `${halo}${ring}${pip(suit, 50, 75, ACE_SIZE[suit])}`;
}

function figureArt(suit, rank) {
  const meta = SUITS[suit];
  const figure = rank === 12 ? 'rey' : rank === 11 ? 'caballo' : 'sota';
  const hold = rank === 12 ? [66, 64] : rank === 11 ? [70, 42] : [68, 70];
  return `
    <rect x="15" y="20" width="70" height="110" rx="4" fill="${meta.robe}" opacity="0.08"/>
    <use href="#f-${figure}" x="18" y="32" width="64" height="96" style="--robe:${meta.robe};--robe-dark:${meta.robeDark}"/>
    ${pip(suit, hold[0], hold[1], 22, suit === 'espada' ? 28 : suit === 'basto' ? 24 : 0)}`;
}

function frame(suit) {
  const meta = SUITS[suit];
  const gaps = [];
  for (let i = 0; i < meta.gaps; i += 1) {
    gaps.push(`<rect x="${26 + i * 7}" y="5.2" width="3.6" height="3.6" fill="#fffcf2"/>`);
    gaps.push(`<rect x="${70.4 - i * 7}" y="141.2" width="3.6" height="3.6" fill="#f4ecd9"/>`);
  }
  return `<rect x="7" y="7" width="86" height="136" rx="3.5" fill="none" stroke="${meta.ink}" stroke-width="1.3" opacity="0.75"/>${gaps.join('')}`;
}

function corner(rank, suit) {
  const ink = SUITS[suit].ink;
  return `
    <text x="13.5" y="24" class="carta-indice" fill="${ink}">${rank}</text>
    <text x="86.5" y="126" class="carta-indice" fill="${ink}" transform="rotate(180 86.5 126)" text-anchor="start">${rank}</text>`;
}

const cache = new Map();

export function cardSVG(card, extraClass = '') {
  const key = `${card.rank}-${card.suit}`;
  let body = cache.get(key);
  if (!body) {
    let art = '';
    if (card.rank === 1) {
      art = aceArt(card.suit);
    } else if (card.rank >= 10) {
      art = figureArt(card.suit, card.rank);
    } else {
      const size = card.rank === 2 ? DUO_SIZE[card.suit] : card.rank === 3 ? Math.round(DUO_SIZE[card.suit] * 0.86) : PIP_SIZE[card.suit];
      art = PIPS[card.rank].map(([x, y], i) => pip(card.suit, x, y, size, tilt(card.suit, i, x))).join('');
    }
    body = `
      <rect class="carta-fondo" x="0.8" y="0.8" width="98.4" height="148.4" rx="8"/>
      ${frame(card.suit)}
      ${corner(card.rank, card.suit)}
      <g class="carta-arte">${art}</g>`;
    cache.set(key, body);
  }
  return `<svg class="carta ${extraClass}" viewBox="0 0 100 150" role="img" aria-label="${card.name || `${card.rank} de ${card.suit}`}">${body}</svg>`;
}

export function backSVG(extraClass = '') {
  return `<svg class="carta dorso ${extraClass}" viewBox="0 0 100 150" role="img" aria-label="carta boca abajo">
    <rect x="0.8" y="0.8" width="98.4" height="148.4" rx="8" fill="#fbf3e2" stroke="#d9cbb0" stroke-width="1"/>
    <rect x="6" y="6" width="88" height="138" rx="5" fill="url(#p-back)"/>
    <rect x="6" y="6" width="88" height="138" rx="5" fill="none" stroke="#e6bf62" stroke-width="1.2"/>
    <rect x="10" y="10" width="80" height="130" rx="3.5" fill="none" stroke="#e6bf62" stroke-width="0.6" opacity="0.7"/>
    <ellipse cx="50" cy="75" rx="21" ry="27" fill="#7a1f24" stroke="#e6bf62" stroke-width="1.4"/>
    <ellipse cx="50" cy="75" rx="16.5" ry="22.5" fill="none" stroke="#e6bf62" stroke-width="0.6" opacity="0.8"/>
    <path d="M50 58 53.5 70 66 75 53.5 80 50 92 46.5 80 34 75 46.5 70Z" fill="#e6bf62" opacity="0.92"/>
    <circle cx="50" cy="75" r="3.2" fill="#7a1f24"/>
  </svg>`;
}

export const SUIT_NAMES = Object.keys(SUITS);
