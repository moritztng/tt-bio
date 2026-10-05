# A contact head with its own loss, on Boltz-2

This example adds an output head to Boltz-2, trains it with a custom loss, and folds a protein it
never saw with the head attached. The model's own weights are not touched. The head reads the
pair representation the trunk computes, on the chip or on a CPU, and runs in plain PyTorch.

```bash
./run.sh                                                # on a host with a Tenstorrent card
ACCEL="--accelerator cpu --no_kernels" ./run.sh         # anywhere else
```

What it does, step by step:

| step | command | what happens |
|---|---|---|
| 0 | `python fetch.py work 1UBQ 1PGA 1CSP 1ENH 1SHG 2CI2 3CHY` | downloads seven PDB entries, writes a single-sequence input and the true Cb-Cb < 8 A contact map for each |
| 1 | `tt-bio predict work/train --head contact_head.py:pair_features` | folds the six training proteins; the `pair_features` head saves each one's pair representation `z` |
| 2 | `python train_head.py work/runs work/truth` | trains `ContactHead` on those with `contact_loss`, writes `contact_head.pt` |
| 3 | `tt-bio predict work/test --head contact_head.py:ContactHead` | folds CheY (3CHY), which the head never saw, and writes `3chy_ContactHead.npz` beside `3chy.cif` |
| 4 | `python score.py ... work/truth` | scores the head's contacts and the fold's own structure against the deposited structure |

`contact_head.py` is the whole modification: two heads and the loss. `contact_loss` is
class-balanced binary cross-entropy over residue pairs at least six apart in sequence, with pairs
24 or more apart counted four times.

Six proteins is a demonstration of the mechanics, not a contact predictor worth using: a real
head wants hundreds of training structures. The point is that each piece is your own Python and
the fold around it is the shipped model, unchanged.
