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
# allowed-tools is used by Claude Code (grants curl/python without prompting);
# Claude Science ignores it and runs code in its own kernel.
allowed-tools:
  - Bash(curl *)
  - Bash(python3 *)
---

# JapanFold — hosted structure prediction & binder design

JapanFold runs Boltz-2 / ESMFold2 / Protenix (structure + affinity) and BoltzGen
(binder design) on Tenstorrent hardware behind a **free public HTTP API**. You
call it as an async job: **submit → poll → download**. No API key is required
(same limits as the web app) and there's no model to install — you're just an
HTTP client of `https://api.japanfold.com`.

Use whatever HTTP tool your environment has — `curl`/Bash, or `httpx`/`requests`
in a Python kernel. If your environment sandboxes network access (e.g. Claude
Science), approve the host **`api.japanfold.com`** when prompted.

## Predict a structure

Submit, then poll until the status is terminal, then read results:

```bash
BASE=https://api.japanfold.com
# 1. submit (input: a bare `sequence`, one `input` FASTA/YAML string, or `targets` list)
JOB=$(curl -s -X POST $BASE/v1/predictions -H 'Content-Type: application/json' \
  -d '{"model":"boltz2","name":"mytarget","sequence":"MKTAYIAKQRQISFVKSHFSRQLEE"}' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')

# 2. poll (jobs take minutes). Tip: add header  Prefer: wait=60  to block up to 60s.
curl -s $BASE/v1/jobs/$JOB      # -> {"status":"queued|running|succeeded|failed", ...}

# 3. when status=succeeded, list artifacts + scores, then download
curl -s $BASE/v1/jobs/$JOB/results          # per-target scores + artifact URLs
curl -sOJ $BASE/v1/jobs/$JOB/archive        # all structures + results.json (zip)
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
  string as `input` instead of `sequence` (`sequences:` with
  `protein`/`dna`/`rna`/`ligand` chains; `properties:` for the affinity head).
- `params` accepts `use_msa_server` (on by default for Boltz-2), `fast`,
  `recycling_steps`, `sampling_steps`, `diffusion_samples`, `output_format`.
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
- No key needed. An optional `Authorization: Bearer <key>` raises the limits.
- Full machine-readable contract: `GET /v1/openapi.json`.
