# `pxd_pagecell.py` code smoke — pc card 0, NOT a page cell

Two short runs whose only job was to prove the harness works end to end before it spends a
benchlock hold on qb2: imports, weight fetch, device open, `model.design`, the CIF write, the
CIF field parsing, the digests, the three-way split and the JSON.

`pxd_smoke_pc0.json` is `--n_step 4 --rounds 2`, `pxd_smoke20_pc0.json` is `--n_step 20
--rounds 3`.

**Neither file is a measurement of anything the page publishes**, for two independent reasons.
The board is wrong: pc is a p150a on a 13x10 grid and every cell in the page's Tenstorrent
column was measured on one AI Processor of qb2's p300c on an 11x10 grid. And the card is the
one root-caused as silently miscomputing ttnn matmuls at a low location-keyed rate, so no
digest recorded here means anything.

What the two runs do establish:

* The harness records what it claims to. 592 tokens, 512 target, 80 binder, 512 conditioned,
  4429 atoms, every coordinate finite, the CIF parsing to 80 residues in one chain A with 321
  atoms — 4 backbone atoms per residue plus the C-terminal OXT the binder carries because it
  is built from a sequence (`write.py`). `split_residual_s` is 0.000, so the three timed leaves
  partition `round_s`.
* `fit_rmsd` converges with step count and the conditioning path is not broken. At 4 steps it
  reads 20568 A, which is unconverged noise, not a defect; at 20 steps the same fixture reads
  48.25 / 41.66 / 32.56 A. Four steps is far below anything the model is run at.
* The per-step cost is linear and the fixed cost is small: 1.019 s of `t_design` at 4 steps and
  1.889 s at 20 gives **54.4 ms per step over a 0.80 s fixed cost**, which puts the shipped
  400-step design at **22.6 s on this board**, host included, against the H200 reference's
  30.8129 s. That is the pre-registered prediction for the qb2 cell, not the cell.
