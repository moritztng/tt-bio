# ProteinMPNN — fixed-backbone sequence design

ProteinMPNN is a 1.66M-parameter message-passing GNN that takes a fixed protein
backbone and returns the amino-acid sequence most likely to fold into it (inverse
folding). It is the sequence-design step run after a backbone generator such as
BoltzGen — the step every design campaign runs many times. Reference:
[dauparas/ProteinMPNN](https://github.com/dauparas/ProteinMPNN) (MIT, code + weights).

## Status

Reference path landed and parity-locked; on-device ttnn port in progress.

- `tt_bio.proteinmpnn` — clean, slim torch reimplementation that loads the official
  `v_48_020.pt` checkpoint unchanged (1,660,485 params, matching the published
  1.66M). Teacher-forced `forward` (the parity gate) and the autoregressive
  `sample` decode loop are both implemented.
- `tests/test_proteinmpnn.py` — asserts per-step log-prob PCC ≥ 0.999 vs the
  reference forward and exact greedy-recovery match on the two bundled test
  backbones (5L33, 6MRR). Passing on qb1.

Measured on the bundled monomers (official `v_48_020`, T=0.1, 3-seed mean for the
sampling number; greedy is deterministic):

| backbone | length | greedy recovery | sampling recovery (T=0.1) |
|----------|--------|-----------------|--------------------------|
| 5L33     | 106    | 0.4623          | 0.4528                   |
| 6MRR     |  68    | 0.5882          | 0.5392                   |

These are within the variance band of the published ~52.4% recovery (which is
averaged over a large benchmark set; two structures are high-variance). They are
the deterministic parity anchors the ttnn port must reproduce, not a new claim.

## Remaining work

- Port the message-passing encoder/decoder to ttnn (dense matmul dominated; the one
  non-matmul primitive is the k=48 neighbour gather, which `ttnn.embedding`
  supports). Verify real-weight PCC ≥ 0.999 on the captured golden I/O.
- Move the autoregressive decode loop on-device (cached per-layer node stack,
  masked decode order) or run hybrid — whichever profiles better. Inference is
  ~0.6–0.9 s / 100 residues on one CPU, so do not over-engineer before profiling.
- Wire a `tt-bio design --backbone x.pdb --sequence-model proteinmpnn` subcommand
  (standalone, bring-your-own-backbone) and as the sequence step following BoltzGen
  backbone generation. Reuse BoltzGen's scheduler/fanout for data-parallel batch
  design across cards.
- Warm throughput (sequences/sec, single- and multi-card).
- LigandMPNN (2.62M params, ligand/nucleotide/metal context encoder) is a documented
  follow-on sharing most of this port.

## Running the reference path

The reference checkpoint and golden fixtures live outside the repo (not committed).
Set them via env or rely on the qb1 defaults:

```bash
PROTEINMPNN_CKPT=~/scratch/ProteinMPNN/vanilla_model_weights/v_48_020.pt \
PROTEINMPNN_GOLDEN_DIR=~/scratch/mpnn_golden \
python -m pytest tests/test_proteinmpnn.py -v
```
