export const $ = (selector, root = document) => root.querySelector(selector);
export const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const ESCAPES = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };

export function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, (ch) => ESCAPES[ch]);
}

export function html(strings, ...values) {
  return strings.reduce((out, part, i) => {
    if (i === 0) return part;
    const value = values[i - 1];
    const text = value && value.__raw ? value.html : Array.isArray(value) ? value.map((v) => (v && v.__raw ? v.html : esc(v))).join('') : esc(value);
    return out + text + part;
  }, '');
}

export function raw(markup) {
  return { __raw: true, html: markup };
}

export const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, Math.max(0, ms)));

export const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

export function pct(p, digits = 0) {
  if (p == null) return '—';
  const value = p * 100;
  if (value > 0 && value < 1 && digits === 0) return '<1%';
  return `${value.toFixed(digits).replace('.', ',')}%`;
}

export function signed(x, digits = 1) {
  if (x == null) return '—';
  const text = Math.abs(x).toFixed(digits).replace('.', ',');
  if (Math.abs(x) < 0.5 * 10 ** -digits) return text;
  return `${x > 0 ? '+' : '−'}${text}`;
}

export function reducedMotion() {
  return window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

export function toast(message, kind = '') {
  const root = document.getElementById('toasts');
  if (!root) return;
  const node = document.createElement('div');
  node.className = `toast ${kind}`;
  node.textContent = message;
  root.append(node);
  setTimeout(() => node.remove(), kind === 'error' ? 5200 : 3200);
}

export function storage(key, fallback = null) {
  try {
    const value = localStorage.getItem(key);
    return value == null ? fallback : JSON.parse(value);
  } catch (error) {
    return fallback;
  }
}

export function store(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch (error) {
    return false;
  }
  return true;
}
