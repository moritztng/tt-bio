# A contact head with its own loss, on Boltz-2

This example adds an output head to Boltz-2, trains it with a custom loss, and folds a protein it
never saw with the head attached. The model's own weights are not touched. The head reads the
pair representation the trunk computes, on the chip or on a CPU, and runs in plain PyTorch.

The example lives in the repository, not in the installed package. Install tt-bio from the same
clone, so the package and the example match: `--head` is newer than the 0.12.0 release on PyPI.
Python 3.10 or 3.12.

```bash
git clone --depth 1 https://github.com/moritztng/tt-bio.git     # 1.1 GB; the full history is 2 GB
cd tt-bio
pip install '.[tenstorrent]'                            # on a host with a Tenstorrent card
pip install .                                           # anywhere else
cd examples/custom_head

./run.sh                                                # on a host with a Tenstorrent card
ACCEL="--accelerator cpu --no_kernels" ./run.sh         # anywhere else
```

The first run downloads the Boltz-2 checkpoints and molecule library into ~/.boltz (7.6 GB) and the
seven PDB entries.

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

## What it printed

Run on a CPU-only host (`--accelerator cpu --no_kernels`, tt-bio at this commit, seed 0). The six
training folds took 59 to 80 s each, and training the head took 4 s (300 epochs, loss 0.694 to
0.128). CheY, which the head never saw:

```
3chy: 128 residues
  |i-j| >= 12: base rate 0.031
    ContactHead     top-L/5 1.000   top-L 0.977
    fold structure  top-L/5 1.000   top-L 1.000
  |i-j| >= 24: base rate 0.027
    ContactHead     top-L/5 1.000   top-L 0.984
    fold structure  top-L/5 1.000   top-L 0.984
```

The same `./run.sh` on one Blackhole chip (qb2, AICLK 1343 to 1350 MHz during each fold) took
24 to 34 s per training fold and 35 s for CheY, with the head attached, once its kernels were
compiled. On a fresh install the first run compiles them as it goes: 88 s for the first training
fold, 20 to 51 s for the rest and 53 s for CheY, 5 min 35 s for the whole script. The scores differ slightly
from the CPU run because the chip computes in lower precision:

```
3chy: 128 residues
  |i-j| >= 12: base rate 0.031
    ContactHead     top-L/5 1.000   top-L 0.992
    fold structure  top-L/5 1.000   top-L 0.984
  |i-j| >= 24: base rate 0.027
    ContactHead     top-L/5 1.000   top-L 0.969
    fold structure  top-L/5 1.000   top-L 0.984
```

Attaching a head does not change the fold. CheY folded again on the same chip without `--head`
gives a byte-identical `3chy.cif` (sha256 `98c3b8ad...8c78d6` both times), and on CPU the same holds
for 1PGA (`d61c2443...0da28e`).

The head ranks contacts nearly as well as the folded structure does, because `z` already carries
the information the trunk uses to place atoms. That is the expected outcome for a head on `z`.

Six proteins is a demonstration of the mechanics, not a contact predictor worth using: a real
head wants hundreds of training structures. The point is that each piece is your own Python and
the fold around it is the shipped model, unchanged.
