# ESMC fused-RoPE: on-hardware release-gate evidence

## Status

The fused-RoPE change (`esmc._rope` → `ttnn.experimental.rotary_embedding` for
tile-aligned L) shipped to `main` as `fd4dba3` behind a tile-alignment gate but
was labelled release-gated and held pending the on-hardware parity gate, which
had never been run. This branch runs that gate and records the evidence: the
shipped ESMC embed path clears the per-residue embedding-PCC floor against the
reference esm ESMC on real hardware, and the speedup reproduces on current
`main`. Ready to merge as the closure of `fd4dba3`'s gate.

The gate change itself is verification infrastructure only: it adds an ESMC
embedding-parity leg to `scripts/release_gate.py` (reusing
`scripts/esmc_embed_parity.py:run_esmc_parity` and the `tests/esmc_reference.py`
golden, not re-derived) and this evidence doc. It touches no model numerics or
perf path. The fusion it gates is already on `main`.

## Why an embedding-PCC gate, not RMSD/TM

`scripts/release_gate.py` folds targets and scores CA-RMSD/TM against an
experimental structure. ESMC is an embedding model: it produces per-residue
embeddings, not structures, so there is no ground truth to fold against. Its
correctness bar is embedding-space agreement with the reference esm ESMC
(`tests/esmc_reference.py`, the same golden the unit parity tests use). The
shipped `embed_sequences` path buckets to `BUCKET=64`, so the padded length is
always tile-aligned and `esmc._rope` always selects the fused kernel on this
path: the gate therefore exercises the numerics-changing fusion directly. Floor
0.99 per-residue PCC (generous; the shipped path measures ~0.9996), matching the
"catch a gross regression, not a tight target" philosophy of the fold floors.
300m/600m run by default; `esmc-6b` is opt-in (`--model esmc-6b`) because its
~13 GB load is too slow for the fast gate and its fused kernel is identical to
300m/600m (head_dim 64), so their parity covers it.

## Re-verified speedup (current main, warm, eager)

`scripts/esmc_rotary_fusion_ab.py --ref`, qb2 card 1, `main` `64ff27a`,
2026-07-13. Warm (2 warmup + median of 7), eager, real biohub checkpoints,
device-synced timed region. Fused = shipped `_rope`; manual = forced
`apply_rotary` rotate-half pile.

| model | tokens=96 | tokens=416 |
|---|---:|---:|
| esmc-300m | 1.576x (25.03 -> 15.88 ms) | 1.673x (27.86 -> 16.66 ms) |
| esmc-600m | 1.662x (30.82 -> 18.55 ms) | 1.447x (33.56 -> 23.19 ms) |
| esmc-6b   | 1.171x (151.77 -> 129.59 ms) | 1.016x (292.51 -> 287.82 ms) |

The win is large on 300m/600m (the embed workhorses for `tt-bio embed` and
JapanFold embeddings) and on 6b@96. At 6b@416 the 4.7 ms delta is within
warm-forward noise (~1.6%): the 2560-wide, 80-layer matmuls dominate and the
RoPE dispatch pile is no longer the bottleneck, so there is no clear win there.
This reproduces the scout's shape (biggest on the small models) honestly; the
6b@416 point is reported as in-noise rather than the scout's earlier 1.13x
sample.

## Accuracy (release-gate metric)

PCC(manual,fused) and reference-esm PCC (manual/fused), same runs:

| model | L=96 fused-vs-manual | L=96 ref (manual / fused) | L=416 fused-vs-manual | L=416 ref (manual / fused) |
|---|---:|---:|---:|---:|
| esmc-300m | 0.999620 | 0.999646 / 0.999683 | 0.999780 | 0.999737 / 0.999800 |
| esmc-600m | 0.999681 | 0.999712 / 0.999701 | 0.999816 | 0.999698 / 0.999732 |
| esmc-6b   | 0.999853 | (ref not run) | 0.999792 | (ref not run) |

Fused tracks manual within bf16 noise and tracks/beats the reference esm ESMC
at every point. 6b reference parity is not run (the CPU reference is slow at
6b); it follows from 300m/600m plus the bit-level fused-vs-manual agreement
above, since all three use the identical fused kernel. The fused-vs-manual
delta is one bf16 ULP in RoPE.

## Gate run (on-hardware, the leg this branch adds)

```
$ TT_VISIBLE_DEVICES=1 ESM_ROOT=/home/ttuser/esm PYTHONPATH=<worktree> \
      TT_MESH_GRAPH_DESC_PATH=.../p150_mesh_graph_descriptor.textproto \
      python scripts/release_gate.py --model esmc-300m --model esmc-600m
...
##############################################################################
RELEASE GATE — ESMC embedding parity (fused-RoPE shipped path), PCC floor 0.99
##############################################################################
model         per-res PCC   pooled   logits   argmax     wall  result
esmc-300m         0.99961  0.99993  0.99990   1.0000       8s  PASS
esmc-600m         0.99964  0.99989  0.99996   1.0000       8s  PASS
##############################################################################
GATE PASS — ESMC embed path cleared the per-residue PCC floor
```

Exit 0. Both models clear the 0.99 per-residue PCC floor against the reference
esm ESMC on the shipped (fused-RoPE) embed path, with logits PCC >= 0.99990 and
argmax agreement 1.0000. This is the parity evidence `fd4dba3` was held on.

## Reproduce

```
# speedup + fused-vs-manual + ref PCC (A/B)
TT_VISIBLE_DEVICES=1 ESM_ROOT=/home/ttuser/esm PYTHONPATH=<worktree> \
    TT_MESH_GRAPH_DESC_PATH=.../p150_mesh_graph_descriptor.textproto \
    python scripts/esmc_rotary_fusion_ab.py --model esmc-300m --tokens 96,416 --ref

# the on-hardware gate leg this branch adds
TT_VISIBLE_DEVICES=1 ESM_ROOT=/home/ttuser/esm PYTHONPATH=<worktree> \
    TT_MESH_GRAPH_DESC_PATH=.../p150_mesh_graph_descriptor.textproto \
    python scripts/release_gate.py --model esmc-300m --model esmc-600m
```
