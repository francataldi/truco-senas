export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

async function request(method, url, body) {
  const options = { method, headers: { Accept: 'application/json' } };
  if (body !== undefined) {
    options.headers['Content-Type'] = 'application/json';
    options.body = JSON.stringify(body);
  }
  let response;
  try {
    response = await fetch(url, options);
  } catch (error) {
    throw new ApiError(0, 'No hay conexión con el servidor. ¿Sigue corriendo scripts/serve.py?');
  }
  if (response.status === 204) return null;
  let data = null;
  try {
    data = await response.json();
  } catch (error) {
    data = null;
  }
  if (!response.ok) {
    let detail = data && data.detail;
    if (Array.isArray(detail)) detail = detail.map((d) => d.msg).join('; ');
    throw new ApiError(response.status, detail || `Error ${response.status}`);
  }
  return data;
}

export const api = {
  agents: () => request('GET', '/api/agents'),
  dictionary: (spec) => request('GET', `/api/dictionary?spec=${encodeURIComponent(spec)}`),
  newGame: (body) => request('POST', '/api/games', body),
  game: (id, since = 0) => request('GET', `/api/games/${id}?since=${since}`),
  act: (id, action, seq) => request('POST', `/api/games/${id}/action`, { action, seq }),
  advance: (id, steps, seq) => request('POST', `/api/games/${id}/advance`, { steps, seq }),
  advice: (id) => request('GET', `/api/games/${id}/advice`),
  remove: (id) => request('DELETE', `/api/games/${id}`),
};
