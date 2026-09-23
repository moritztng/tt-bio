# of3t-msafwd pre-registration

Written and committed before any arm of this row ran. Inputs to it: `state/of3t-msaamp.md`, the
code of `tt_bio/openfold3_msa_embedder.py`, `tt_bio/tenstorrent.py:PairformerLayer` and upstream
0.4.3 `latent/msa_module.py` + `base_blocks.py`. No new number.

## The quantity

At the msa_module boundary the real-block forward rel_l2 is 8.176e-03 ours, 2.311e-03 upstream
0.4.3 bf16 (CPU autocast). Excess squared error X = 8.176^2 - 2.311^2 = 61.5e-06. A carrier's
share is the part of X it removes when only it is brought to upstream's precision:
share(C) = (8.176e-3^2 - E_ours_without_C^2) / X. The bisect runs on the 64-token crop, where
msaamp measured 8.350e-03 against 2.311e-03 (padding carries none of the gap); the 384 A/A is
re-taken first and must reproduce 8.176e-03.

## What differs structurally

Upstream under CPU autocast rounds matmul inputs to bf16 but keeps every residual add, every
LayerNorm and the residual stream itself in fp32. Ours holds `z` and `m` in bf16 on device
between every op: 4 blocks x 6 `z` residual adds (opm, tri_mul_out, tri_mul_in, tri_att_start,
tri_att_end, pair_transition) = 24 bf16 roundings of `z`, 6 of `m`, plus the bf16 output. One
bf16 rounding costs ~1.6e-03 relative rms. msaamp's bf16-inputs arm moved upstream only
2.311 -> 2.390e-03, so one rounding of `z_in` reads ~0.6e-03 at `z_out`: `z` grows through the
module and early roundings are diluted. Later states are closer to |z_out|, so 24 roundings
predict roughly 4e-03 to 7e-03 of residual-storage error before any op error. That is the size of
the whole gap.

## Prediction (a genuine disjunction, probabilities sum to 1)

H-R, 0.55: **residual storage carries it, not an op.** Keeping `z` (and `m`) in fp32 between ops
on device, with every op still fed its bf16 copy, brings ours to <= 4.0e-03 (share >= 0.78), AND
rounding the residual to bf16 after every add in upstream's own bf16 chain raises upstream to
>= 6.0e-03. The per-family local errors are then each within 1.5x of upstream's own.

H-F, 0.25: **one family's local op error carries it** (share >= 0.5 from one family's
teacher-forced local error excess). Candidates in order: pair_transition (0.10; top of the
gradient excess table), tri_att_start or tri_att_end (0.10 together), msa_att_row / opm /
msa_transition (0.05 together). tri_mul_out / tri_mul_in are not candidates: their gradients
are MORE accurate than upstream's.

H-S, 0.20: **spread.** No carrier, residual storage included, reaches share 0.5; the top one
carries 0.2 to 0.5.

## Falsifiers

H-R is falsified if the device fp32-residual arm leaves ours >= 6.5e-03 (share < 0.37), or if the
bf16-residual emulation leaves upstream < 4.0e-03. Between those and the H-R bands is a partial
H-R, reported as its share. H-F is falsified for a family if its teacher-forced local injected
error excess is < 0.5 X. H-S is falsified by any carrier >= 0.5.

## Closure check

Teacher-forced local injected errors (each site fed the float64 input, its update compared with
the float64 update, absolute error normalised by |z_out|) are summed in quadrature per side,
residual rounding included. If that sum does not reproduce each side's chain error within 1.5x,
propagation is not ~identity and the shares are quoted from the substitution arms only.

## Reference's own error first

Every per-site and per-op number for ours is printed beside upstream bf16's own at the same site,
and beside the pure bf16-storage floor of that site's output (the f64 value rounded to bf16),
before any site of ours is called a defect.
