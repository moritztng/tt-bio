---
name: japanfold
description: >-
  Predict 3D biomolecular structures and binding affinity (Boltz-2, ESMFold2,
  Protenix) and design de-novo binders/proteins (BoltzGen) via JapanFold — a
  free, public, Tenstorrent-accelerated HTTP API. Use to fold a protein or
  complex, co-fold a protein with a ligand and get affinity, design
  nanobody/antibody/peptide/miniprotein binders against a target, or turn a
  sequence into a PDB/mmCIF structure. No API key or local GPU needed.
when_to_use: >-
  When the user wants to fold/predict a protein or complex structure, estimate
  protein–ligand binding affinity, or design binders against a target — and a
  hosted service is fine (no local model to run).
license: Apache-2.0
category: biomodels
metadata:
  third_party:
    - kind: service
      name: JapanFold API
      provider: JapanFold
      info_url: https://japanfold.com
# allowed-tools is a Claude Code convenience (grants curl/python without a
# prompt); other harnesses ignore it and use their own execution/permission model.
allowed-tools:
  - Bash(curl *)
  - Bash(python3 *)
---

# JapanFold — hosted structure prediction & binder design

JapanFold runs Boltz-2 / ESMFold2 / Protenix (structure + affinity) and BoltzGen
(binder design) on Tenstorrent hardware behind a **free public HTTP API**. You
call it as an async job — **submit → poll → download** — over plain HTTPS against
`https://api.japanfold.com`. No API key, no model to install, no local GPU.

Works from any agent/harness: use `curl` (Bash) or your language's HTTP client
(`httpx`/`requests`, `fetch`, `net/http`, …) — whatever your environment has.
If your environment sandboxes network egress (e.g. Claude Science), approve the
host **`api.japanfold.com`** when prompted.

## Predict a structure

Submit → poll until `status` is terminal → read results:

```bash
BASE=https://api.japanfold.com
# 1. submit — input is a bare `sequence`, one `input` FASTA/YAML string, or a `targets` list
JOB=$(curl -s -X POST $BASE/v1/predictions -H 'Content-Type: application/json' \
  -d '{"model":"boltz2","name":"mytarget","sequence":"MKTAYIAKQRQISFVKSHFSRQLEE"}' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')

# 2. poll (a small fold is usually done in well under a minute). Tip: add header
#    `Prefer: wait=60` to the GET to block until the job finishes (up to 60s).
curl -s $BASE/v1/jobs/$JOB          # -> {"status":"queued|running|succeeded|failed", ...}

# 3. once status=succeeded: scores + artifact URLs, then download the bundle
curl -s   $BASE/v1/jobs/$JOB/results
curl -sOJ $BASE/v1/jobs/$JOB/archive          # zip: structures + results.json
```

**Multi-chain complexes** (e.g. insulin's A+B chains) go in the `input` YAML —
one `protein` entry per chain, not the bare `sequence` field:

```bash
curl -s -X POST $BASE/v1/predictions -H 'Content-Type: application/json' -d '{
  "model":"boltz2","name":"human-insulin",
  "input":"sequences:\n  - protein: {id: A, sequence: GIVEQCCTSICSLYQLENYCN}\n  - protein: {id: B, sequence: FVNQHLCGSHLVEALYLVCGERGFFYTPKT}\n"
}'
```

Python-kernel equivalent (Claude Science, notebooks):

```python
import time, httpx
BASE = "https://api.japanfold.com"
job = httpx.post(f"{BASE}/v1/predictions",
                 json={"model": "boltz2", "sequence": "MKT..."}).json()
while job["status"] not in ("succeeded", "failed", "canceled"):
    time.sleep(5)
    job = httpx.get(f"{BASE}/v1/jobs/{job['id']}").json()
res = httpx.get(f"{BASE}/v1/jobs/{job['id']}/results").json()
```

- **Models:** `boltz2` (default; MSA + ligands + affinity), `esmfold2`,
  `esmfold2-fast` (single-sequence, fastest), `protenix-v2`.
- For complexes / protein–ligand affinity / multiple chains, pass a **Boltz YAML**
  string as `input` (`sequences:` with `protein`/`dna`/`rna`/`ligand` chains;
  `properties:` for the affinity head).
- `params`: `use_msa_server` (on by default for Boltz-2), `fast`, `recycling_steps`,
  `sampling_steps`, `diffusion_samples`, `output_format`.
- `GET /v1/models` lists every model, protocol, parameter, and the current limits.

## Design binders (BoltzGen)

```bash
curl -s -X POST $BASE/v1/designs -H 'Content-Type: application/json' \
  -d '{"protocol":"nanobody-anything","spec":"<YAML design spec>","params":{"num_designs":10}}'
```

Protocols: `protein-anything`, `peptide-anything`, `nanobody-anything`,
`antibody-anything`, `protein-small_molecule`, `protein-redesign`. Poll the same
way; `/v1/jobs/{id}/results` returns the ranked designs.

## Reading results

`GET /v1/jobs/{id}/results` gives `ready`, an `artifacts` list (each with a `url`),
and — for a prediction — per-target `rows` (`confidence_score`, `complex_plddt`,
`iptm`, affinity fields); for a design, the ranked `designs`. Pass lines mirror
Boltz-2: interface `iptm` > 0.5, fold `complex_plddt` > 0.7. Download a single
structure from its artifact `url`, or the whole bundle from `…/archive`.

## Limits & notes

- Free public demo caps (same as the web app): **≤ 1024 residues/structure,
  ≤ 10 chains & ligands/complex, ≤ 10 structures/run, ≤ 10 designs/request**,
  plus per-IP rate limits. Over a cap → `400`; at capacity → `429` (respect
  `Retry-After`). Numeric params are clamped to range.
- Errors are RFC 9457 problem+json (`title`, `detail`).
- No key needed; an optional `Authorization: Bearer <key>` raises the limits.
- Full machine-readable contract: `GET /v1/openapi.json`.
- **If a request ever returns HTTP `403` with Cloudflare error `1010`**, that's
  edge-level bot filtering of your HTTP client, not an API error — retry the same
  request with a browser-like `User-Agent` header
  (`Mozilla/5.0 … Chrome/124.0 Safari/537.36`).
