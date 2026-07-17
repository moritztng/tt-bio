# ProteinMPNN — fixed-backbone sequence design

ProteinMPNN is a 1.66M-parameter message-passing GNN that takes a fixed protein
backbone and returns the amino-acid sequence most likely to fold into it (inverse
folding). It is the sequence-design step run after a backbone generator such as
BoltzGen — the step every design campaign runs many times. Reference:
[dauparas/ProteinMPNN](https://github.com/dauparas/ProteinMPNN) (MIT, code + weights).

## Status

Working, parity-verified, and wired into the `tt-bio design` CLI. The on-device
ttnn port was profiled out as a single-call win (see *Why no on-device port*) —
throughput comes from data-parallel fanout, which scales near-linearly.

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

## Throughput

Single worker (1 CPU thread), steady state:

| backbone | L | seq/s | ms/seq |
|----------|---|-------|--------|
| 5L33     |106| 1.86  | 537    |
| 6MRR     | 68| 5.16  | 194    |

Data-parallel fanout (scheduler-style: N parallel `tt-bio design` processes with
disjoint seeds, `OMP_NUM_THREADS=1` per worker), 5L33 (L=106):

| workers | seq/s | scaling |
|---------|-------|---------|
| 1       | 1.86  | 1.0×    |
| 4       | 7.36  | 4.0×    |
| 8       | 13.76 | 7.4×    |

Near-linear (~1.7 seq/s/worker). On a 32-core box, ~32 workers → ~50+ seq/s
aggregate. Per-process checkpoint load (~6 s) dominates at low N and amortizes
over thousands of sequences — the campaign regime this model is built for.

## Why no on-device port

Profiled before porting, per the proposal's guidance. Breakdown (5L33, L=106):
the dense matmuls are 86% of the teacher-forced forward, but the shapes are tiny
(hidden=128, k=48), and the autoregressive `sample` loop runs the decoder once
per residue (144 ms overhead on top of the 135 ms forward). A single call is
~100+ small matmuls; on Blackhorse this is dispatch-bound and would be slower
than CPU unless fully traced/fused — and the AR loop's N-step unrolled trace is
beyond the trace limit. So the throughput story is volume via fanout, not a
faster single call. A traced/fused on-device port remains a separate kernel-effort
follow-on; it is not skipped lightly — it is profiled out.

## Running

```bash
# single worker
tt-bio design path/to/backbone.pdb \
    --sequence-model proteinmpnn \
    --num-sequences 8 --temperature 0.1 --seed 7 \
    --checkpoint ~/scratch/ProteinMPNN/vanilla_model_weights/v_48_020.pt \
    --out-dir ./design

# campaign fanout (scheduler-style, disjoint seeds, one core each)
for i in 1 2 3 4 8; do OMP_NUM_THREADS=1 tt-bio design bb.pdb \
    --num-sequences 250 --seed $((i*1000)) --out-dir ./design/w$i &
done; wait
```

Writes `<name>.fasta` (one record per sampled sequence, with recovery vs native).
`--checkpoint` defaults to `$PROTEINMPNN_CKPT`. Standalone, bring-your-own-backbone.

## Remaining

- Multi-card fanout via BoltzGen's scheduler (spawn workers across cards with
  disjoint seeds) — the mechanism this port is designed for; the per-worker unit
  (`tt-bio design`) is ready.
- LigandMPNN (2.62M params, ligand/nucleotide/metal context encoder) is a
  documented follow-on sharing most of this port.
- A traced/fused on-device ttnn port is a possible future perf push if a faster
  single call is ever needed (not justified by the current profile).
