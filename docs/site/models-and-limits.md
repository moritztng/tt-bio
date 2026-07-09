# Models & limits

The live, machine-readable version of this page is `GET /v1/models` — it returns
the model list, every parameter with its default/range, the design protocols, and
the current numeric limits. The tables below mirror it; if they ever disagree,
**trust the live endpoint**.

```bash
curl -s https://api.japanfold.com/v1/models
```

`GET /v1/health` is a plain liveness check (`{"status":"ok","api_version":"…"}`)
if you just need to know the service is up.

## Models

| `id` | Name | MSA | Ligands | Nucleic acids | Affinity | Constraints | PAE/PDE |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|
| `boltz2` | Boltz-2 | default on | ✓ | ✓ | ✓ | ✓ | — |
| `esmfold2` | ESMFold-2 | optional | — | — | — | — | — |
| `esmfold2-fast` | ESMFold-2 Fast | single-seq only | — | — | — | — | — |
| `protenix-v2` | Protenix-v2 | default on | ✓ | ✓ | — | — | ✓ |

- **Boltz-2** — most capable; choose it when in doubt, or whenever you need
  ligands, nucleic acids, affinity or constraints.
- **ESMFold-2 / ESMFold-2 Fast** — language-model folding, protein chains only.
  Fast, lightweight; `-fast` is always single-sequence, for screening many
  sequences at once.
- **Protenix-v2** — AlphaFold3-family (Pairformer + atom diffusion), strong at
  antibody–antigen. Emits PAE/PDE and per-atom pLDDT. For affinity, use Boltz-2.

## Prediction parameters

Sent as `params` on `POST /v1/predictions`. Out-of-range values are clamped.

| Key | Type | Default | Range | Notes |
|---|---|---|---|---|
| `use_msa_server` | bool | `true` | — | Build an MSA. Required for Boltz-2/Protenix; optional for ESMFold-2. |
| `fast` | bool | `true` | — | Higher throughput, may be slightly less accurate. |
| `recycling_steps` | int | `3` | 1–10 | More can improve accuracy, slower. |
| `sampling_steps` | int | `200` | 10–500 | Diffusion steps per structure. |
| `diffusion_samples` | int | `1` | 1–5 | Structures generated per target. |
| `output_format` | enum | `cif` | `cif`, `pdb` | Structure file format. |

## Design protocols & parameters

Protocols for `POST /v1/designs`: `protein-anything`, `peptide-anything`,
`nanobody-anything`, `antibody-anything`, `protein-small_molecule`,
`protein-redesign`. See [Designs](designs.md) for what each does.

| Key | Type | Default | Range | Notes |
|---|---|---|---|---|
| `num_designs` | int | `10` | 1–10 | Binders to generate before filtering. |
| `budget` | int | `10` | 1–10 | Top ranked designs to keep after filtering. |
| `fast` | bool | `true` | — | Higher throughput, may be slightly less accurate. |

## Limits (free tier)

This is a free public demo on shared compute, so inputs and concurrency are
capped. The full platform has no such limits.

**Per structure / complex**

| Limit | Value |
|---|---|
| `max_residues` (per structure) | 1024 |
| `max_chains_per_complex` | 10 |
| `max_ligands_per_complex` | 10 |
| `max_constraints_per_complex` | 20 |
| `max_complexes` (structures per run) | 10 |
| `max_content_chars` (per input string) | 50000 |

**Design**

| Limit | Value |
|---|---|
| `max_designs` | 10 |
| `max_budget` | 10 |

**Parameter ceilings**

| Limit | Value |
|---|---|
| `max_recycling_steps` | 10 |
| `max_sampling_steps` | 500 |
| `max_diffusion_samples` | 5 |

**Concurrency & rate**

| Limit | Value |
|---|---|
| `max_active_jobs` (service-wide) | 64 |
| `max_active_jobs_per_ip` | 8 |
| `max_active_jobs_per_session` | 3 |
| `max_submits_per_min` (service-wide) | 12 |
| `max_submits_per_min_per_ip` | 40 |
| `max_retained_jobs` | 200 |

**Runtime / stall guards**

| Limit | Value |
|---|---|
| `max_runtime_predict_s` | 1500 |
| `max_runtime_design_s` | 2700 |
| `max_stall_s` (predict) | 600 |
| `max_stall_design_s` | 1200 |

Exceed a size cap → `400`. At capacity / over a rate limit → `429` with
`Retry-After`. See [Errors](errors.md).
