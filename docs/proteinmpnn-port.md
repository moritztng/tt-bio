# ProteinMPNN — fixed-backbone sequence design

ProteinMPNN is a 1.66M-parameter message-passing GNN that takes a fixed protein
backbone and returns the amino-acid sequence most likely to fold into it (inverse
folding). It is the sequence-design step run after a backbone generator such as
BoltzGen — the step every design campaign runs many times. Reference:
[dauparas/ProteinMPNN](https://github.com/dauparas/ProteinMPNN) (MIT, code + weights).

## Status

Reference path landed, parity-locked, and wired into the `tt-bio design` CLI. The
on-device ttnn port is the remaining perf work (see *Remaining*).

- `tt_bio/proteinmpnn.py` — clean, slim torch reimplementation that loads the
  official `v_48_020.pt` checkpoint unchanged (1,660,485 params = published 1.66M).
  Teacher-forced `forward` (the parity gate) and the autoregressive `sample`
  decode loop (cached per-layer node stack) are both implemented.
- `tt_bio/proteinmpnn_data.py` — slim PDB parser + featurizer (all-chains-designed
  path) so the CLI is standalone.
- `tests/test_proteinmpnn.py` — parity vs the reference forward (PCC ≥ 0.999, exact
  greedy-recovery match) plus an end-to-end design sanity test. 4/4 passing on qb1.

## Parity and recovery (measured on qb1, official `v_48_020`, no fabrication)

| backbone | L | log-prob PCC vs ref | greedy recovery | sampling recovery (T=0.1) |
|----------|---|--------------------|-----------------|--------------------------|
| 5L33     |106| ≥0.999 (asserted)  | 0.4623          | 0.45–0.47                |
| 6MRR     | 68| ≥0.999 (asserted)  | 0.5882          | 0.54–0.59                |

Greedy recovery matches the reference exactly (to <5e-4). The sampling band sits
within the variance of the published ~52.4% (a large-set average; two structures
are high-variance) — a sanity anchor, not a new benchmark claim.

## Throughput (single CPU, 4 threads)

| backbone | L | seq/s | ms/seq |
|----------|---|-------|--------|
| 5L33     |106| 3.36  | 298    |
| 6MRR     | 68| 5.16  | 194    |

Faster per-call than the proposal's ~0.6–0.9 s/100-res CPU estimate. The throughput
story is data-parallel fanout at campaign scale (thousands of sequences per
backbone), reusing BoltzGen's scheduler — not single-call latency. Multi-card TT
throughput is measured after the on-device port.

## Running

```bash
tt-bio design path/to/backbone.pdb \
    --sequence-model proteinmpnn \
    --num-sequences 8 --temperature 0.1 --seed 7 \
    --checkpoint ~/scratch/ProteinMPNN/vanilla_model_weights/v_48_020.pt \
    --out-dir ./design
```

Writes `<name>.fasta` (one record per sampled sequence, with recovery vs the native).
`--checkpoint` defaults to `$PROTEINMPNN_CKPT`. Standalone, bring-your-own-backbone.

## Remaining

- Port the message-passing encoder/decoder to ttnn (dense-matmul dominated; the one
  non-matmul primitive is the k=48 neighbour gather, supported by `ttnn.embedding`).
  Verify real-weight PCC ≥ 0.999 on the captured golden I/O. Watch: no native
  `ttnn.gelu`; `ttnn.layer_norm` exists; fp32 matmul accumulates in bf16.
- AR decode loop on-device (cached `h_V_stack` + masked decode order) or hybrid —
  profile before over-engineering.
- Multi-card fanout via BoltzGen's scheduler; warm multi-card throughput (seq/s).
- LigandMPNN (2.62M params, ligand/nucleotide/metal context encoder) is a documented
  follow-on sharing most of this port.
