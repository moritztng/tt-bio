# b2z2-elision-bh-measure — does the atom key gather elision survive onto Blackhole?

VERDICT: BLOCKED
BRANCH: wk/b2z2-elision-bh-measure
ARCH: BH
CARD: qb2 card 0 (`TT_VISIBLE_DEVICES=0`), one Blackhole processor of the p300c, 11x10 grid.

## PRE-REGISTRATION — written and pushed before any Blackhole number existed

PREDICTED: I expect the lever to survive, at **1.008x on the whole 512 aa fold** (band
1.005x-1.013x), and **1.02x-1.05x on the sampler**. Bit-exact both arms.

The arithmetic behind that number, so it is falsifiable rather than a hunch:

* The lever is entirely inside the diffusion step. On Blackhole the sampler is **5.28-5.32 s of
  the 20.079 s published cell**, i.e. **26.4 %** of the fold (CONTEXT.md §2-CORRECTION).
* If Wormhole's **1.04694x** sampler ratio transferred unchanged, Amdahl gives
  `1 / (0.736 + 0.264/1.04694)` = **1.0120x** on the fold. That is *larger* than Wormhole's own
  1.00926x fold number, because the Blackhole sampler is a bigger share of a shorter fold.
* I discount that. Blackhole's 110-core grid and higher clock make each deleted movement program
  cheaper in absolute ms, while the replacement's surviving ROW_MAJOR round trip on 1.2 MB is
  bandwidth-bound and does not shrink by the same factor. Wave 2's record is that WH ratios shrink
  on BH, never grow. So: **1.008x**, a third off the clean Amdahl transfer.

FALSIFIER: a paired fold ratio at or below the cell's A/A floor of **1.00229x** refutes the
prediction and closes the wave's last untested bit-exact lever at 1.000x. That is the outcome the
brief expects and I am not going to stretch away from it. I also commit in advance to reporting the
**worst** of the per-rep paired ratios, not only the median: a lever this small is only real if it
wins in most reps, not if one lucky rep carries the median.

Second, separable prediction: **bit-exact holds**. The gather is a pure index permutation and the
host gate demands `torch.equal` against the rebuilt one-hot before the fast path is taken, so an
arithmetic difference is not the risk. The risk is Blackhole fold nondeterminism, which has bitten
this campaign twice. I will therefore check within-arm sha256 agreement separately from across-arm:
if the OFF arm is not self-identical across its own reps, that is a finding about the box, not about
the lever, and it will be reported as one.

MEASURED: pending — this section is the pre-registration only.
PARITY: pending.
DEFICIT-SECONDS: pending.
TILE-MOVEMENT-DELTA: pending.
CHEAT-CHECK: nothing in this row changes the model's work. 200 sampling steps, 3 recycles, full
35-row MSA depth, 1 sample, seed 0, templates off. The lever replaces the program that assembles a
key window; every atom still attends to the same 128 keys with the same weights.

## Protocol, fixed before the run

`perf/b2z2_elision/fold_ab.py` (the same harness that produced the Wormhole number, so the two
architectures are compared through one instrument), which imports its protocol and cfg from
`perf/b2x-flag-levers/ab_flag_levers.py`. `perf/size512/fixtures/cdk2x2_512.yaml` + its fixed 35-row
a3m, 3 recycles, 200 sampling steps, 1 sample, seed 0, templates off, timed at `predict_one`, one
cold fold per arm discarded, arms interleaved `off, on, off, on` inside one process so each rep has
two base positions. Under `benchlock`. Polarity: `_ATOM_SHIFT_GATHER_OFF = not
env_flag("TT_BIO_ATOM_SHIFT_GATHER", True)`, so the lever ships ON on this branch and the base arm
is the one-hot matmul that main runs today.
