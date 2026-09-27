# of3t-p10instr pre-registration: grading "same updates" on a multi-draw held-out loss

Written and committed before any A/B arm (TF7, B2trunk, I_TFs1, I_D12, I_D12s1, X12b) has been
scored on this instrument. The only multi-seed numbers in existence when this was committed are
the noise runs N3a/N3b on the start weights and the two 40-step device checkpoints L40G/L40Gs1,
none of which is an arm. The rules below are coded in `grade.py`, committed with this file, and
the verdict is whatever that script prints.

## Why the old instrument cannot grade

`trainarm.py::_evaluate` scores each held-out target on ONE diffusion noise draw
(`denoise_draw(eval_seed)`). Measured this row (`out/N1_repro.json`, `out/N2_order.json`):

* same weights, same seed, one process: the start checkpoint reads 7ohe 10.762624 on the first
  call and 18.871191 on every call after it;
* X12b step 11 reads 7fb8 4.717948 when it is the first weights a process scores and 2.720573
  when the start weights ran first, the other three targets agreeing to 3e-5;
* inside one warm state the eval is exactly repeatable, and scoring inside `install()` or on the
  shipped path gives the same digits.

So a single-draw target loss is a coin that a ~1e-5 perturbation flips, and a one-draw A/B
measures the coin.

## The instrument

For a checkpoint X, `evalnoise.py` calls `_evaluate` at the 32 eval seeds S = {0, ..., 31}
(seed 20260926, the old single draw, is scored too and reported, never graded). All weights of
one A/B are scored in ONE process, after one discarded warm-up call, in the order the grade
lists them. Per target t, m_t(X) = mean over S of the loss. The held-out score is
M(X) = mean over the four targets of m_t(X).

The MEAN over draws is graded, not the median: for a two-mode target the median is itself a
coin once the mode probability is near one half, while the mean moves smoothly with it.

For two checkpoints A and B scored at the same seeds, d_s = mean_t [L_t(A, s) - L_t(B, s)] and
Delta(A, B) = mean_s d_s. The draws are paired, so the noise the two share cancels.
Half-width h = 1.96 * sd(d_s) / sqrt(32) + H, where H is the process-history term: the largest
|M| difference between N3a and N3b over the three weights both score.

## The floor and the bar

The accepted variation is re-training with another data order. For each design the training-seed
floor is F = |Delta(device seed 0, device seed 1)|, scored on the same instrument in the same
process as the arms. The bar is T = F / 3, the ratio of the standing 512 aa structure bar to its
seed floor (0.60 A against 1.84 A).

## The two A/Bs

1. **Two steps**, the design `of3t-p10trainout` pre-registered: A = TF7 (fixed device path,
   seed 0), B = B2trunk (exact trunk, seed 0), floor pair TF7 vs I_TFs1 (fixed device, seed 1).
   Score order: base, TF7, B2trunk, I_TFs1.
2. **Twelve steps**: A = I_D12 (fixed device, seed 0, saved at step 11), B = X12b (exact trunk,
   seed 0, step 11), floor pair I_D12 vs I_D12s1. Score order: base, I_D12, X12b, I_D12s1.

Each design grades:

* **PASS** if |Delta(A, B)| + h <= T;
* **FAIL** if |Delta(A, B)| - h > T;
* **UNRESOLVED** otherwise, and also whenever the floor itself is not resolved
  (F <= h_F, the floor pair's own half-width), because then there is no bar to hold it to.

## The verdict

* **GO**: both designs PASS.
* **NO-GO**: the twelve-step design FAILS.
* **PARTIAL**: anything else.

## What is not graded, and why

The training-loss curve (device minus exact over 12 steps, `of3t-p10trainfix`) is already
measured, so it cannot be pre-registered here, and its only floor on hand, another seed's
curve, trains on different batches, so the two losses are not comparable step by step. It is
reported beside the verdict and does not enter it.
