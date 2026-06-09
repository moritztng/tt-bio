# ai& Bio platform

A lightweight web platform on top of **tt-bio** — a drug-discovery front end for
Boltz-2, ESMFold-2 and BoltzGen running on Tenstorrent. It exposes everything
tt-bio can do (structure prediction, binding-affinity prediction, and de-novo
binder/nanobody/antibody design) with progressive disclosure: simple for new
users, full power for advanced ones, with in-browser 3D structure viewing.

## Architecture

```
React SPA (Vite)  ──fetch──▶  Flask API (app.py)  ──▶  JobManager (jobs.py)
   src/, built to static/         /api/*                  one worker, FIFO queue
                                                              │ subprocess
                                                              ▼
                                                      tt-bio predict / gen run
                                                      (the real engine, on TT)
```

- **No new engine code.** Each job shells out to the real `tt-bio` CLI, so served
  results are identical to running tt-bio by hand. There is no second code path.
- **Stateless-ish.** Jobs live as directories under the workspace
  (`~/.aiand-bio/jobs` by default); metadata is a `meta.json` per job, so the
  server can restart and still list history.
- **No auth / no DB** by design — minimal first iteration. The API boundary is
  clean enough to later fold into ai&'s console (orgs / credits / keys).

## Running

```bash
pip install -e '.[platform]'        # adds flask + flask-cors
tt-bio serve                        # http://0.0.0.0:8080
# or: python -m tt_bio.platform --port 8080 --workspace /data/jobs
```

`tt-bio serve` serves the prebuilt frontend from `static/`. On the serving host
you typically run `tt-bio worker --connect ...` against your Galaxy, or let the
default in-process scheduler use the local devices.

## Developing the frontend

```bash
cd tt_bio/platform/frontend
npm install
npm run dev      # Vite on :5173, proxies /api to Flask on :8080 (run `tt-bio serve` too)
npm run build    # writes the production bundle into ../static (tracked in git)
```

## Layout

```
tt_bio/platform/
  app.py          Flask app + routes (/api/*, SPA)
  jobs.py         JobManager: queue, subprocess runner, result parsing
  catalog.py      models, protocols, tunable params, examples (drives the UI)
  static/         built React app (committed so `tt-bio serve` needs no Node)
  frontend/       React + Vite source
```

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/catalog` | models, protocols, params, examples |
| GET | `/api/jobs` | list jobs |
| POST | `/api/jobs` | submit a job |
| GET | `/api/jobs/<id>` | job status + parsed results |
| GET | `/api/jobs/<id>/log` | raw run log |
| GET | `/api/jobs/<id>/structure/<relpath>` | a result structure file (CIF/PDB) |
| POST | `/api/jobs/<id>/cancel` | cancel a running/queued job |
| DELETE | `/api/jobs/<id>` | delete a job and its outputs |
