const PYODIDE_VERSION = '0.27.8';
const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const BUNDLE_URL = new URL('../py/truco.zip', import.meta.url);
const APP_DIR = '/home/pyodide/app';

let starting = null;

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = src;
    script.async = true;
    script.onload = resolve;
    script.onerror = () => reject(new Error(`No se pudo bajar ${src}`));
    document.head.appendChild(script);
  });
}

function progress(text, fraction) {
  const box = document.querySelector('#pano .vacio-mesa');
  if (!box) return;
  box.innerHTML = `
    <div class="carga" role="status">
      <div class="carga-titulo">${text}</div>
      <div class="carga-barra"><span style="width:${Math.round(fraction * 100)}%"></span></div>
      <div class="carga-nota">Todo corre en tu navegador: Python, el motor del truco y las redes neuronales de los agentes. La primera vez tarda unos segundos.</div>
    </div>`;
  const pill = document.getElementById('mode-text');
  if (pill) pill.textContent = 'Cargando';
}

async function start() {
  progress('Cargando Python en el navegador…', 0.08);
  await loadScript(`${PYODIDE_URL}pyodide.js`);
  const pyodide = await window.loadPyodide({ indexURL: PYODIDE_URL });
  progress('Cargando numpy…', 0.4);
  const bundle = fetch(BUNDLE_URL).then((r) => {
    if (!r.ok) throw new Error(`No se pudo bajar el motor (${r.status})`);
    return r.arrayBuffer();
  });
  await pyodide.loadPackage('numpy');
  progress('Bajando el motor y los agentes…', 0.65);
  pyodide.unpackArchive(await bundle, 'zip', { extractDir: APP_DIR });
  progress('Preparando la mesa…', 0.85);
  pyodide.runPython(`
import os, sys
sys.path.insert(0, "${APP_DIR}")
os.chdir("${APP_DIR}")
from demo.local import setup, handle
setup()
`);
  const handle = pyodide.globals.get('handle');
  progress('Listo', 1);
  return handle;
}

export function ready() {
  if (!starting) {
    starting = start().catch((error) => {
      starting = null;
      throw error;
    });
  }
  return starting;
}

export async function localRequest(method, url, body) {
  const handle = await ready();
  await new Promise((resolve) => setTimeout(resolve, 0));
  const out = JSON.parse(handle(method, url, body === undefined ? null : JSON.stringify(body)));
  return out;
}
