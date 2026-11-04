const LOOKS = [
  { skin: '#f0c9a2', shade: '#d9a983', hair: '#3a2a1f', shirt: 'var(--team-0)', style: 'short' },
  { skin: '#d8a27b', shade: '#bb8461', hair: '#8f8a84', shirt: 'var(--team-1)', style: 'bald' },
  { skin: '#e9bc93', shade: '#cf9f77', hair: '#4b3121', shirt: 'var(--team-0)', style: 'boina' },
  { skin: '#c48a5f', shade: '#a8704a', hair: '#24180f', shirt: 'var(--team-1)', style: 'curly' },
];

export const GESTURES = ['cejas', 'guiño', 'boca a la derecha', 'boca a la izquierda', 'labio mordido', 'lengua afuera', 'ojos cerrados', 'beso'];

function hair(look) {
  switch (look.style) {
    case 'short':
      return `<path d="M27 44C25 27 36 18 50 18S75 26 73 44C70 34 62 29 50 29S31 33 27 44Z" fill="${look.hair}"/>
        <path d="M33 30C40 25 48 24 56 26" stroke="#5a4535" stroke-width="1.4" fill="none" stroke-linecap="round" opacity="0.6"/>`;
    case 'bald':
      return `<path d="M27.5 48C26 41 27 36 29.5 33C30.5 39 30.5 44 31 49Z" fill="${look.hair}"/>
        <path d="M72.5 48C74 41 73 36 70.5 33C69.5 39 69.5 44 69 49Z" fill="${look.hair}"/>
        <path d="M39 25C43 23 47 22.5 51 23" stroke="#f3d1b4" stroke-width="2" fill="none" stroke-linecap="round" opacity="0.55"/>`;
    case 'boina':
      return `<path d="M28 42C27 35 31 31 35 30L65 30C69 31 73 35 72 42C68 37 60 35 50 35S32 37 28 42Z" fill="${look.hair}"/>
        <path d="M23 31C24 20 38 13 52 13S78 19 77 29C71 27 64 26 50 26S29 28 23 31Z" fill="#26313d"/>
        <path d="M23 31C29 28 40 27 50 27S71 28 77 29" stroke="#3b4a5a" stroke-width="2" fill="none"/>
        <circle cx="51" cy="12.4" r="2.2" fill="#26313d"/>`;
    default:
      return `<path d="M25 52C20 40 24 22 38 17C44 13 56 13 62 17C76 22 80 40 75 52C73 43 70 37 66 34C61 37 55 33 50 34C45 33 39 37 34 34C30 37 27 43 25 52Z" fill="${look.hair}"/>
        <circle cx="31" cy="24" r="5" fill="${look.hair}"/><circle cx="41" cy="17" r="5.5" fill="${look.hair}"/>
        <circle cx="53" cy="15.5" r="5.5" fill="${look.hair}"/><circle cx="65" cy="19" r="5.5" fill="${look.hair}"/>
        <circle cx="72" cy="28" r="5" fill="${look.hair}"/>`;
  }
}

function eyes(gesture) {
  const open = (x, big) => `<ellipse cx="${x}" cy="46" rx="${big ? 3.2 : 2.7}" ry="${big ? 4 : 3.3}" fill="#2a1f18"/><circle cx="${x + 0.9}" cy="44.8" r="0.9" fill="#fff" opacity="0.85"/>`;
  const shut = (x) => `<path d="M${x - 4} 46.5Q${x} 49.5 ${x + 4} 46.5" stroke="#2a1f18" stroke-width="1.8" fill="none" stroke-linecap="round"/>`;
  if (gesture === 1) return open(41) + shut(59);
  if (gesture === 6) return shut(41) + shut(59);
  return open(41, gesture === 0) + open(59, gesture === 0);
}

function brows(gesture, look) {
  const color = look.style === 'bald' ? '#6f6a64' : look.hair;
  if (gesture === 0) {
    return `<path d="M35 33.5Q41 29.5 47 33" stroke="${color}" stroke-width="2.4" fill="none" stroke-linecap="round"/>
      <path d="M53 33Q59 29.5 65 33.5" stroke="${color}" stroke-width="2.4" fill="none" stroke-linecap="round"/>`;
  }
  if (gesture === 1) {
    return `<path d="M35.5 39Q41 36 46.5 38.5" stroke="${color}" stroke-width="2.4" fill="none" stroke-linecap="round"/>
      <path d="M53.5 40.5Q59 39.5 64.5 41" stroke="${color}" stroke-width="2.4" fill="none" stroke-linecap="round"/>`;
  }
  return `<path d="M35.5 39Q41 36 46.5 38.5" stroke="${color}" stroke-width="2.4" fill="none" stroke-linecap="round"/>
    <path d="M53.5 38.5Q59 36 64.5 39" stroke="${color}" stroke-width="2.4" fill="none" stroke-linecap="round"/>`;
}

function mouth(gesture) {
  switch (gesture) {
    case 2:
      return '<path d="M46 62Q54 64 60.5 57.5" stroke="#7b3b2c" stroke-width="2.3" fill="none" stroke-linecap="round"/>';
    case 3:
      return '<path d="M54 62Q46 64 39.5 57.5" stroke="#7b3b2c" stroke-width="2.3" fill="none" stroke-linecap="round"/>';
    case 4:
      return `<path d="M42.5 61.5Q50 65 57.5 61.5" stroke="#b2574a" stroke-width="3" fill="none" stroke-linecap="round"/>
        <rect x="45.2" y="57.6" width="9.6" height="4" rx="1.2" fill="#fffaf2" stroke="#b9ab98" stroke-width="0.5"/>
        <path d="M50 57.8V61.4" stroke="#d4c8b8" stroke-width="0.6"/>`;
    case 5:
      return `<path d="M43 58.5Q50 57 57 58.5Q56 64.5 50 64.5T43 58.5Z" fill="#5a2018"/>
        <path d="M45.6 61.5Q50 60 54.4 61.5V65.5Q54.4 70.5 50 70.5T45.6 65.5Z" fill="#e26d7d" stroke="#b94a5a" stroke-width="0.8"/>
        <path d="M50 62.2V67.6" stroke="#b94a5a" stroke-width="0.8" stroke-linecap="round"/>`;
    case 7:
      return `<ellipse cx="50" cy="61" rx="3.4" ry="3" fill="#c55a5a" stroke="#8f3434" stroke-width="1"/>
        <path d="M47.6 61H52.4" stroke="#8f3434" stroke-width="0.8"/>
        <path d="M70 26c0-2.2 3-2.8 3.8-0.6.8-2.2 3.8-1.6 3.8.6 0 2.4-3.8 4.6-3.8 4.6s-3.8-2.2-3.8-4.6Z" fill="#e0555f"/>`;
    case 6:
      return '<path d="M44 60.5Q50 62.5 56 60.5" stroke="#7b3b2c" stroke-width="2.2" fill="none" stroke-linecap="round"/>';
    case 1:
      return '<path d="M43 60Q51 64.5 58 58.5" stroke="#7b3b2c" stroke-width="2.3" fill="none" stroke-linecap="round"/>';
    case 0:
      return '<ellipse cx="50" cy="61" rx="3.6" ry="3.4" fill="#5a2018"/>';
    default:
      return '<path d="M43 59.5Q50 64 57 59.5" stroke="#7b3b2c" stroke-width="2.3" fill="none" stroke-linecap="round"/>';
  }
}

function extras(look, gesture) {
  if (look.style === 'bald') {
    const shift = gesture === 2 ? 3 : gesture === 3 ? -3 : 0;
    return `<path d="M${38 + shift} 56C${43 + shift} 53 ${48 + shift} 54 ${50 + shift} 55.5C${52 + shift} 54 ${57 + shift} 53 ${62 + shift} 56C${57 + shift} 57.5 ${53 + shift} 58 ${50 + shift} 57C${47 + shift} 58 ${43 + shift} 57.5 ${38 + shift} 56Z" fill="#6f6a64"/>`;
  }
  if (look.style === 'curly') {
    return `<circle cx="41" cy="46" r="7" fill="none" stroke="#2b2b2b" stroke-width="1.6" opacity="0.9"/>
      <circle cx="59" cy="46" r="7" fill="none" stroke="#2b2b2b" stroke-width="1.6" opacity="0.9"/>
      <path d="M48 45.5Q50 44 52 45.5" stroke="#2b2b2b" stroke-width="1.4" fill="none"/>`;
  }
  return '';
}

export function faceSVG(seat, gesture = null, extraClass = '') {
  const look = LOOKS[seat % LOOKS.length];
  const g = gesture == null ? -1 : gesture;
  const cheeks = g === 5 || g === 7 || g === 0
    ? '<circle cx="35.5" cy="54" r="3.6" fill="#e58a7a" opacity="0.35"/><circle cx="64.5" cy="54" r="3.6" fill="#e58a7a" opacity="0.35"/>'
    : '';
  return `<svg class="cara ${extraClass}" viewBox="0 0 100 100" role="img" aria-hidden="true">
    <circle cx="50" cy="50" r="49" class="cara-fondo"/>
    <path d="M14 100C16 82 30 74 50 74S84 82 86 100Z" fill="${look.shirt}"/>
    <path d="M42 70H58V80Q50 86 42 80Z" fill="${look.shade}"/>
    <ellipse cx="26.5" cy="49" rx="4" ry="6" fill="${look.shade}"/>
    <ellipse cx="73.5" cy="49" rx="4" ry="6" fill="${look.shade}"/>
    <ellipse cx="50" cy="47" rx="23" ry="26" fill="${look.skin}"/>
    ${hair(look)}
    ${brows(g, look)}
    ${eyes(g)}
    <path d="M50 47.5Q47.6 54.5 49.6 55.6Q51 56.2 52.4 55.4" stroke="${look.shade}" stroke-width="1.7" fill="none" stroke-linecap="round"/>
    ${cheeks}
    ${extras(look, g)}
    ${mouth(g)}
  </svg>`;
}
