# b2z2-sampler-union-wh — PREDICTION, written before the first device measurement of this row

Written 2026-09-12 ~23:2x CEST on branch `wk/b2z2-sampler-union-wh`, after the ancestry and
source-diff analysis of the six branches and BEFORE any number from this row's own hardware
existed. Not edited after the first measurement.

## P0 — the population is not six levers

From `git merge-base --is-ancestor` and the tt_bio/ source diffs, before any device run:

* `wk/b2z2-step-program-fusion` is an **ancestor of four** of the other five branches
  (`step-fusion-next-sites`, `atom-window-next`, `step-matmul-group`, `step-binaryng-fusion`).
* `wk/b2z2-step-matmul-group` changes **zero bytes of `tt_bio/`** beyond `step-program-fusion`.
  Its own state doc says the lever it built "turned out to be a duplicate of another row's and was
  reverted". It is not a lever.
* `TT_BIO_ATOM_SHIFT_GATHER` (`layout-op-elision`) and `TT_BIO_ATOM_KEY_WINDOW`
  (`step-program-fusion`) **replace the same one-hot gather matmul at the same call site with the
  same shift-slice-concat construction**. They are one lever built twice.
* `TT_BIO_MAC_FUSE` (`step-binaryng-fusion`) is its own row's NO-GO: 1.6 % slower, best arm
  1.00044x inside a 1.00122x floor. It is not in the union.

So the union is **three levers on ONE site**: the gather elision (two names), `TT_BIO_ATOM_L1`,
`TT_BIO_ATOM_KV_PREPROJ`. All three act inside `AttentionPairBias`'s atom branch.
**PREDICTED: the site sets are NOT disjoint, and this is the reason the discount will be large.**

## P1 — product of singles

    gather (AKW)      1.07444x on the step
    ATOM_L1           1.08461x on the step
    KV_PREPROJ        1.01566x on the step, on top of the gather
    product                                 1.18360x

The brief's table would also multiply `TT_BIO_ATOM_SHIFT_GATHER` at 1.04694x, giving 1.23915x.
**PREDICTED: that 1.04694x is a pure double count of the 1.07444x and must not appear in any stack
projection.**

## P2 — the measured union step ratio

**PREDICTED 1.12x - 1.16x, point estimate 1.138x.**

Why below the product: `TT_BIO_ATOM_L1`'s 1.08461x was measured with the gather OFF, so a
substantial part of it is L1 residency applied to the reshape/permute/matmul chain that the gather
lever **deletes outright**. Those saved bytes cannot be saved twice. `KV_PREPROJ` shrinks a matmul
that L1 has already made cheap.

**FALSIFIER, pre-registered: if the measured union is more than 3 % below 1.18360x, the wave's
stack projection is wrong. At my point estimate the discount is 3.8 %, so I am predicting the
falsifier TRIPS.** Band on the discount: 2 % - 5.5 %.

## P3 — the fold

The WH 512 aa fold is ~41.2 s with a ~10.0 s sampler (24.3 % share, `b2z2-layout-op-elision`'s
`fold_ab_512_whglx_c4.json`). Amdahl on a 1.138x sampler gives 1.0255x.

**PREDICTED fold ratio 1.018x - 1.038x, point estimate 1.026x**, paired interleaved, n >= 6.
A/A floor predicted under 1.010x on a box whose loadavg is not mine to control.

## P4 — parity, and where I expect it to fail

The brief's bar is `torch.equal` on the step output. I do not predict a clean sweep:

* **base vs gather: PREDICTED NOT bit-exact.** The shipped path's gather is a `bfloat4_b` one-hot
  **matmul**, and `b2z2-step-program-fusion` measured it missing the exact gather by 0.03125 max
  abs. The window reproduces the gather's definition bit for bit; the matmul does not. So the
  difference is the SHIPPED path's fidelity loss, not the lever's, and the honest statement is
  "the lever is exact and the thing it replaces is not".
* **gather(AKW) vs gather(SHG): PREDICTED bit-exact, max abs 0.0.** Same transform.
* **gather vs gather+KV_PREPROJ: PREDICTED bit-exact, max abs 0.0** (a linear is per-row).
* **base vs ATOM_L1: PREDICTED bit-exact, max abs 0.0** (a memory config moves bytes, not values).
* **UNION vs base: PREDICTED non-zero at exactly the gather's 0.03125**, and **UNION vs gather-only
  bit-exact**.
* **At the FOLD level PREDICTED the CIF sha256 matches base**, because `TT_BIO_ATOM_KEY_WINDOW`
  writes the published digest `a91aa44441f0d9c5` on the Blackhole cell.

## P5 — the two gather constructions, head to head

`_atom_shift_gather` cuts the source at the real window count before the ROW_MAJOR shift;
`_atom_window_plan` does not look at the matrix and pads the whole bucket.
**PREDICTED: within 2 % of each other on the step at 512 aa**, because the bucket has no tail at
this size (`windows == K == 140`). If `windows < K`, PREDICTED the shape-gated construction is
**not** bit-exact against the matmul while the matrix-proved one is.

## P6 — what does not move

**PREDICTED zero effect outside `Diffusion.__call__`.** None of the three touches the Pairformer
or MSA blocks, so the trunk wall is predicted within its own A/A floor across all arms.
