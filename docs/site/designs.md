# Designs

`POST /v1/designs` designs de-novo binders against a target with
[BoltzGen](https://github.com/jwohlwend/boltz). It returns a **job** to poll,
exactly like [predictions](jobs.md). When it succeeds, the results hold the
ranked designs.

## Request

```bash
curl -s -X POST https://api.japanfold.com/v1/designs \
  -H 'Content-Type: application/json' \
  -d '{
    "protocol":"nanobody-anything",
    "name":"my-nanobodies",
    "spec":"sequences:\n  - protein: {id: A, sequence: MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ}\n",
    "params":{"num_designs":10,"budget":10,"fast":true}
  }'
```

- **`spec`** (required) — a YAML design spec: the target plus the binder request.
  Same YAML dialect as the Boltz/BoltzGen inputs.
- **`protocol`** — what to design (see below). 
- **`name`** — optional label.
- **`params`** — see below.

## Protocols

| `protocol` | Designs |
|---|---|
| `protein-anything` | De-novo mini-protein binder against any target. |
| `peptide-anything` | Short peptide binder. |
| `nanobody-anything` | Single-domain antibody / nanobody (VHH). |
| `antibody-anything` | Antibody binder. |
| `protein-small_molecule` | Protein binder with a binding-affinity step. |
| `protein-redesign` | Re-design residues of an existing binder. |

## Parameters

| Param | Type | Default | Range | Notes |
|---|---|---|---|---|
| `num_designs` | int | 10 | 1–10 | Binders to generate before filtering. |
| `budget` | int | 10 | 1–10 | Top ranked designs to keep after filtering. |
| `fast` | bool | true | — | Higher throughput, may be slightly less accurate. |

(Free-tier ranges; see [Models & limits](models-and-limits.md).)

Submits accept the same `Idempotency-Key` and `Prefer: wait` headers as
predictions — see [Predictions](predictions.md#retrying-safely-idempotency-key).

## Reading designs

Poll `GET /v1/jobs/{id}` until `succeeded`, then `GET /v1/jobs/{id}/results`.
The results carry the ranked designs and downloadable structure artifacts;
`GET /v1/jobs/{id}/archive` gives everything as one zip. See
[Jobs → Results](jobs.md#results). A full worked example is in
[Examples → Binder design](examples.md#binder-design).
