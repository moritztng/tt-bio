# Distance to GO, restated on the corrected 0.4.3 denominator

Pass 111. The pass-96 map was computed on the 0.5.0 model's 4,147 parameters and said
"94.4 % of the proof mass is closed or one named measurement away". Two things were wrong with
that: the denominator, and the conflation of **unblocked** with **measured**.

## Corrected shares (0.4.3, 4,170 parameters)

Source: `of3t-rebase`'s `perf/of3t_rebase/reach_by_norm_043.json`, measured on a card, whose
`reference.sha256` is the same `1d4ea922…95cc4` the orchestrator hashed for A13.

| section | tensors | share of squared gradient norm | (superseded 0.5.0 figure) |
|---|---|---|---|
| `diffusion_module` | 761 | **89.211 %** | 91.208 |
| `pairformer_stack` | 2,736 | **5.828 %** | 3.156 |
| `aux_heads` | 244 | **2.843 %** | 4.265 |
| `msa_module` | 227 | **1.240 %** | 0.893 |
| `input_embedder` | 98 | **0.801 %** | 0.400 |
| `msa_module_embedder` + others | — | **0.077 %** | 0.078 |

## The distinction that matters

**Forward verified**, D23 closed and inside the 5.0e-02 per-tensor bar:

- `diffusion_module` — 89.211 %, one-block bisection every boundary inside bar, depth ladder
  2.168e-02 over 24 blocks
- pairformer blocks 0 and 23 — ~0.32 %
- **total ≈ 89.53 %**

**Per-parameter GRADIENT passing at §3d bars** — which is what instrument A and the charter
actually require:

- pairformer block 0 at crop 64, median 1.2136e-02 — PASS
- pairformer block 23 at crop 64, median 1.9191e-02 — PASS
- everything else — **no passing reading**
- **total ≈ 0.32 %**

So the campaign has verified a great deal about the **forward** and almost nothing yet about the
**gradient**. Saying "94.4 % closed or one measurement away" blurred exactly that line.

## How the 0.32 % is estimated, and its caveat

`replay_vs_r0.json` put block 0 at 0.086 % of a 3.156 % trunk, i.e. **2.72 % of the trunk**.
Rescaled to the corrected 5.828 % trunk that is **~0.159 % per block**, so blocks 0 and 23
together are **~0.32 %**. This is an estimate: the 48 blocks are not exactly equal and block 23's
share is not separately measured. The order is what is robust — **sub-1 %**, not double digits.

## What is open, by size

| | share | status |
|---|---|---|
| `diffusion_module` | 89.211 % | forward closed; gradient gated on `diffcap043` → A18 → instrument A. **Capture running.** |
| `pairformer_stack` | 5.828 % | blocks 0 and 23 PASS at crop 64; **block 47 fails at 2.0129e-01, 52/52 over bar** — a depth-graded residual D23 does not explain |
| `aux_heads` | 2.843 % | unmeasured; owned by `of3t-auxheads`, held on `DEPENDS_ON` |
| `msa_module` + `input_embedder` | **2.041 %** | unmeasured and **UNOWNED** — this bucket nearly doubled under the corrected denominator (was 1.29 %) and no row has it |
| others | 0.077 % | negligible |

## Standing bounds, unchanged

Crops 640 and 768 are out of reach on one p300c (~49–52 GB and ~70–75 GB against 34.22 GB), so
the demonstrated training scope is **crop 384, one of upstream's four stage configs**. §6 coverage
is 7 of 8 terms with `bond` firing nowhere. And nothing here bears on stability over a real
100k-step run, precision drift, or convergence to the published weights.
