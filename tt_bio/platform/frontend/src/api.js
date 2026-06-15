// Tiny fetch wrapper around the ai& Drug Discovery API.

async function json(r) {
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { msg = (await r.json()).error || msg; } catch { /* ignore */ }
    throw new Error(msg);
  }
  return r.json();
}

export const api = {
  catalog: () => fetch("/api/catalog").then(json),
  jobs: () => fetch("/api/jobs").then(json),
  job: (id) => fetch(`/api/jobs/${id}`).then(json),
  submit: (body) =>
    fetch("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(json),
  cancel: (id) => fetch(`/api/jobs/${id}/cancel`, { method: "POST" }).then(json),
  remove: (id) => fetch(`/api/jobs/${id}`, { method: "DELETE" }).then(json),
  logUrl: (id) => `/api/jobs/${id}/log`,
  structureUrl: (id, rel) => `/api/jobs/${id}/structure/${rel}`,
  archiveUrl: (id) => `/api/jobs/${id}/archive`,
};
