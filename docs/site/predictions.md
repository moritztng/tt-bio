# Predictions

`POST /v1/predictions` predicts the 3D structure — and, with Boltz-2, the
binding affinity — of a protein or complex. It returns a **job** to poll (see
[Jobs](jobs.md)).

## Three ways to specify input

Pick whichever fits. You provide **one** of these:

### 1. `sequence` — a single protein chain (simplest)

```bash
curl -s -X POST https://api.japanfold.com/v1/predictions \
  -H 'Content-Type: application/json' \
  -d '{"model":"boltz2","name":"myprotein","sequence":"MKTAYIAKQRQISFVKSHFSRQLEE"}'
```

### 2. `input` — one FASTA or Boltz YAML string

Use this for **complexes, multiple chains, ligands, nucleic acids, affinity and
constraints**. The string is a full [Boltz](https://github.com/jwohlwend/boltz)
YAML (or FASTA). Note the `\n` newlines when embedding it in JSON:

```bash
# Human insulin — two protein chains (A + B)
curl -s -X POST https://api.japanfold.com/v1/predictions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"boltz2","name":"human-insulin",
    "input":"sequences:\n  - protein: {id: A, sequence: GIVEQCCTSICSLYQLENYCN}\n  - protein: {id: B, sequence: FVNQHLCGSHLVEALYLVCGERGFFYTPKT}\n"
  }'
```

### 3. `targets` — a list, to fold many inputs in one job

Each target is `{ "content": "<FASTA or YAML>", "name": "<optional>" }`.

```bash
curl -s -X POST https://api.japanfold.com/v1/predictions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"esmfold2-fast",
    "targets":[
      {"name":"a","content":">a\nMKTAYIAKQRQISFVKSHFSRQLEE"},
      {"name":"b","content":">b\nGIVEQCCTSICSLYQLENYCN"}
    ]
  }'
```

## Choosing a model

Set `model` (default `boltz2`). See [Models & limits](models-and-limits.md) for
the full capability matrix.

| Model | Use it for |
|---|---|
| `boltz2` | The default and most capable. Proteins, DNA, RNA, ligands, **affinity**, constraints. Uses an MSA by default. |
| `esmfold2` | Language-model folding, proteins only. Fast; MSA optional. |
| `esmfold2-fast` | ESMFold-2 tuned for throughput — always single-sequence. For screening many sequences. |
| `protenix-v2` | AlphaFold3-family (Pairformer + atom diffusion). Complexes, PAE/PDE output. No affinity. |

Only Boltz-2 does affinity, constraints and potentials. ESMFold-2 is protein-only.

## Co-folding with a ligand + affinity (Boltz-2)

Add a `ligand` chain (SMILES or CCD code) and a `properties: affinity` block
naming the binder chain:

```bash
curl -s -X POST https://api.japanfold.com/v1/predictions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"boltz2","name":"prot-ligand",
    "input":"sequences:\n  - protein: {id: A, sequence: MKTAYIAKQRQISFVKSHFSRQLEE}\n  - ligand: {id: L, smiles: \"CC(=O)Oc1ccccc1C(=O)O\"}\nproperties:\n  - affinity: {binder: L}\n"
  }'
```

The results then include affinity fields alongside the structure and confidence
scores (see [Jobs → Results](jobs.md#results)).

## Parameters

Pass a `params` object. Values are clamped to their allowed range (see
[Models & limits](models-and-limits.md) for defaults, ranges and per-model
applicability).

| Param | Type | Notes |
|---|---|---|
| `use_msa_server` | bool | Build an MSA. On by default for Boltz-2 and Protenix-v2; optional for ESMFold-2. |
| `fast` | bool | Higher throughput, slightly lower precision. |
| `recycling_steps` | int | More can improve accuracy at the cost of speed. |
| `sampling_steps` | int | Diffusion steps per structure. |
| `diffusion_samples` | int | Number of structures to generate per target. |
| `output_format` | enum | `cif` (default) or `pdb`. |

```bash
curl -s -X POST https://api.japanfold.com/v1/predictions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"boltz2","sequence":"MKTAYIAK...",
    "params":{"use_msa_server":true,"diffusion_samples":3,"output_format":"pdb"}
  }'
```

> **MSA and privacy.** With `use_msa_server` on (the Boltz-2 / Protenix-v2
> default), your sequence is sent to an external MSA server for the alignment
> step. To fold strictly single-sequence, set `use_msa_server: false`.

## Waiting inline: the `Prefer: wait` header

By default a submit returns immediately with a job to poll. Add
`Prefer: wait=<seconds>` (also accepted on the `POST /v1/predictions` itself, or
on a **`GET /v1/jobs/{id}`** poll) to block until the job finishes or the
timeout elapses — turning poll loops into one call for short jobs. `wait` alone
holds for 25s; `wait=N` holds up to `N` seconds, capped at 60:

```bash
curl -s -H 'Prefer: wait=60' https://api.japanfold.com/v1/jobs/$JOB
```

## Retrying safely: `Idempotency-Key`

Send an `Idempotency-Key: <unique>` header on a create. A retried submit with
the same key (and same caller) returns the original job instead of launching a
duplicate — useful when a client retries on a dropped connection:

```bash
curl -s -X POST https://api.japanfold.com/v1/predictions \
  -H 'Content-Type: application/json' -H 'Idempotency-Key: run-42' \
  -d '{"model":"boltz2","sequence":"MKTAYIAK..."}'
```

## Response

`202`-style body is a **Job** object — `id`, `status`, `kind: "predict"`,
`model`, timestamps and a `links` map. Poll `links.self` (or
`/v1/jobs/{id}`) and read `/v1/jobs/{id}/results` when `results_ready` is true.
See [Jobs](jobs.md) for the full lifecycle and result shape.
