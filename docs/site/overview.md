# JapanFold API

Fold proteins, co-fold complexes with ligands (and get binding affinity), and
design de-novo binders — over a **free, public, keyless HTTP API**. Boltz-2,
ESMFold-2 and Protenix-v2 for structure prediction, BoltzGen for binder design.
No API key, no local GPU, nothing to install.

```
Base URL   https://api.japanfold.com
Contract   https://api.japanfold.com/v1/openapi.json   (OpenAPI 3.1, the source of truth)
```

## The model: submit → poll → download

Everything is an **async job**. You submit work, get back a job `id`, poll it
until the `status` is terminal, then download the results.

```
POST /v1/predictions   or   POST /v1/designs      →  { "id": "...", "status": "running", ... }
GET  /v1/jobs/{id}                                 →  poll until status is succeeded/failed/canceled
GET  /v1/jobs/{id}/results                          →  scores + a list of downloadable artifacts
GET  /v1/jobs/{id}/archive                          →  everything as one .zip
```

Statuses: `queued` → `running` → `succeeded` | `failed` | `canceled`.

> Server artifacts are retained only temporarily. Download what you need and save
> it locally.

## Fold your first protein in 3 calls

```bash
BASE=https://api.japanfold.com

# 1. submit — a bare `sequence` is the simplest input
JOB=$(curl -s -X POST $BASE/v1/predictions \
  -H 'Content-Type: application/json' \
  -d '{"model":"boltz2","name":"myprotein","sequence":"MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"}' \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')

# 2. poll until done. `Prefer: wait=60` blocks up to 60s so it often returns finished.
curl -s -H 'Prefer: wait=60' $BASE/v1/jobs/$JOB

# 3. once status=succeeded: read scores, then download the structures
curl -s $BASE/v1/jobs/$JOB/results
curl -s $BASE/v1/jobs/$JOB/archive -o myprotein.zip && unzip -oq myprotein.zip -d myprotein
```

That's the whole workflow. Everything else — complexes, ligands, affinity,
binder design, model and parameter choice — is a variation on these calls.

## Where to go next

Read the guides in order, or jump to what you need:

- **[Authentication](authentication.md)** — keyless by default; a Bearer key raises the limits.
- **[Predictions](predictions.md)** — input shapes, models, co-folding, affinity, params.
- **[Designs](designs.md)** — BoltzGen binder design.
- **[Jobs](jobs.md)** — polling, listing, cancel/delete, results, logs, artifacts, archive.
- **[Models & limits](models-and-limits.md)** — the model list, every parameter, and the caps.
- **[Errors](errors.md)** — the problem+json shape and the status codes.
- **[Examples](examples.md)** — end-to-end fold, co-fold+affinity and design in curl and Python.
- **[The JapanFold skill](skill.md)** — fold and design straight from your AI agent.

## A note on network egress

Some environments sandbox outbound HTTP. If so, allow the host
**`api.japanfold.com`**. If a request ever returns HTTP `403` with Cloudflare
error `1010`, that is edge bot-filtering of your HTTP client, not an API error —
retry with a browser-like `User-Agent` header.
