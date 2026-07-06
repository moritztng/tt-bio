---
name: japanfold
description: >
  Predict biomolecular structures and binding affinity (Boltz-2, ESMFold2,
  Protenix) and design de-novo binders/proteins (BoltzGen) via the JapanFold
  API — a hosted, Tenstorrent-accelerated service. Reach for this to fold a
  protein or complex, co-fold a protein with a ligand and get affinity, or
  design nanobody/antibody/peptide/miniprotein binders against a target,
  without provisioning a local GPU. JapanFold is a pure HTTP service; you call
  it as an async job API (submit → poll → download).
license: Apache-2.0
category: biomodels
metadata:
  third_party:
    - kind: service
      name: JapanFold API
      provider: JapanFold
      info_url: https://japanfold.com
---

# JapanFold — hosted structure prediction & binder design

JapanFold runs Boltz-2 / ESMFold2 / Protenix (structure + affinity) and
BoltzGen (binder design) on Tenstorrent hardware behind an async HTTP API. You
are a **pure HTTP client of `BASE_URL`** — there is no local model, no GPU to
provision, no weights to download. Pick JapanFold over the local `boltz` /
`esmfold2` skills when you want the managed hosted service (Tenstorrent compute,
one API for prediction + affinity + design) rather than running a model in this
kernel.

## Reaching the API (network)

JapanFold lives at a single host (production: `https://japanfold.com`; a
self-hosted deployment has its own URL). Two supported ways to call it:

- **As a registered model endpoint (recommended for repeated use).** Register
  once, then every `compute_provider` cell gets `BASE_URL` + `INFER_API_KEY`
  preloaded and network egress scoped to JapanFold via the sandbox proxy:

  ```python
  host.model_endpoints.register(
      name="japanfold-service",
      url="https://japanfold.com",          # or http://127.0.0.1:8080 for a local deployment
      credential="NVIDIA_API_KEY",           # family credential NAME; value = your JapanFold key
      skill="japanfold",
      live="/v1/health",
  )
  ```
  Then dispatch inference cells with `compute_provider(provider="japanfold-service", code=...)`.

- **As a plain external API.** From a kernel cell, call it with `httpx`/`requests`.
  The sandbox scopes network egress, so the JapanFold host must be approved first
  (an approval card appears; approve `japanfold.com`). Send the key as
  `Authorization: Bearer <JAPANFOLD_API_KEY>`.

Get an API key from https://japanfold.com/account (format `jf_live_…`).

## Predict a structure

```python
import os, time, httpx
BASE = os.environ.get("BASE_URL", "https://japanfold.com")
H = {"Authorization": f"Bearer {os.environ.get('INFER_API_KEY') or JAPANFOLD_API_KEY}"}

# Submit. Input can be a single `sequence`, one `input` FASTA/YAML string, or a
# list of `targets`. Boltz-2 is the default model (MSA + ligands + affinity).
job = httpx.post(f"{BASE}/v1/predictions", headers=H, json={
    "model": "boltz2", "name": "mytarget",
    "sequence": "MVTPEGNVSLVDESLLVGVTDEDRAVRS...",
}).json()

# Poll (jobs take minutes). Or send header `Prefer: wait=60` on the GET to block
# up to 60s and return terminal results in one call.
while job["status"] not in ("succeeded", "failed", "canceled"):
    time.sleep(5)
    job = httpx.get(f"{BASE}/v1/jobs/{job['id']}", headers=H).json()
assert job["status"] == "succeeded", job.get("error")

# Results: per-target scores + downloadable structures.
res = httpx.get(f"{BASE}/v1/jobs/{job['id']}/results", headers=H).json()
```

For complexes, protein–ligand affinity, multiple chains, or constraints, pass a
Boltz YAML string as `input` instead of `sequence` (same schema the local
`boltz` skill documents: `sequences:` with `protein`/`dna`/`rna`/`ligand`
chains, optional `properties:` for the affinity head).

- Models: `boltz2` (default), `esmfold2`, `esmfold2-fast` (single-sequence,
  fastest), `protenix-v2`. `params` accepts `use_msa_server` (on by default for
  Boltz-2), `fast`, `recycling_steps`, `sampling_steps`, `diffusion_samples`,
  `output_format`. `GET /v1/models` lists everything.

## Design binders (BoltzGen)

```python
job = httpx.post(f"{BASE}/v1/designs", headers=H, json={
    "protocol": "nanobody-anything",   # or protein-/peptide-/antibody-anything, protein-small_molecule, protein-redesign
    "spec": open("design.yaml").read(),
    "params": {"num_designs": 10},
}).json()
# poll as above, then read ranked designs from /v1/jobs/{id}/results
```

## Reading results

`GET /v1/jobs/{id}/results` returns `ready`, an `artifacts` list (each with a
`url` under `/v1/jobs/{id}/artifacts/…`), and for a prediction the per-target
`rows` (`confidence_score`, `complex_plddt`, `iptm`, affinity fields), or for a
design the ranked `designs`. Community pass lines mirror Boltz-2: interface
`iptm` > 0.5, fold `complex_plddt` > 0.7. Download a single structure from its
artifact `url`, or the whole bundle from `GET /v1/jobs/{id}/archive` (a zip of
CIF/PDB structures + `results.json`). Fetch structures into the workspace and
open the `.cif` files to inspect or visualize the fold.

## Notes

- Everything is async: submit returns a job with `status: queued`; poll
  `/v1/jobs/{id}` until terminal. `Prefer: wait[=seconds]` (≤60) collapses fast
  jobs into one call. Retries are safe with an `Idempotency-Key` header.
- Errors are RFC 9457 problem+json (`title`, `detail`); `401` = bad/missing key,
  `429` = at capacity (respect `Retry-After`).
- The full contract is at `GET /v1/openapi.json`.
