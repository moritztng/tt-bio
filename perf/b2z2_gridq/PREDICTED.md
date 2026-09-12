# b2z2-grid-qchunk-unify — pre-registered, before the first grep and the first device run

Written 2026-09-13 by `b2z2-grid-qchunk-unify`, committed before the constant audit and before any
measurement on any card. The brief asks for two numbers up front: how many other grid-blind
constants I expect to find, and the Blackhole step ratio.

## P0 — how many other grid-blind constants are on this path

A "grid-blind constant" here means a literal that decides how much work a device op gets per core,
or how many pieces the work is cut into, and that is written down as a number rather than derived
from `COMPUTE_GRID_MAIN`. `_capped_sdpa_chunk_size` is the instance already convicted.

**Predicted: 5 to 9 such constants in `tt_bio/`, of which 1 to 3 carry measurable exposure**
(>= 1.01x on some shipped shape) and the rest are either already grid-gated, dead at shipped
shapes, or bounded by something other than occupancy.

Named in advance, from reading the SDPA picker's neighbourhood only:

1. `SDPA_CHUNK_MAX` (the 256 cap itself) — convicted, this row's lever.
2. The trimul chunk **budget**, whose own comment calls it "a 130-core calibration". A budget
   calibrated on one core count is the same defect with an L1 term attached.
3. `SEQ_LEN_MORE_CHUNKING` (608) — a length threshold that switches chunking behaviour with no
   grid term; a 72-core part and a 110-core part should not switch at the same length.
4. `TRIANGLE_MULT_CHUNK_SIZE` — a floor the clamp loops stop at.
5. `_dividing_sdpa_chunk_size`'s `cap/2` floor — a fraction of a grid-blind cap is grid-blind.

I expect at least one more in the MSA or atom path that I have not read yet, and I expect at least
one of the five above to turn out **already grid-aware** (`_IS_SMALL_GRID` exists, so somebody has
partly done this) — the audit's value is the exposure column, not the count.

**Falsifier for P0:** if the audit finds 0 or 1 further grid-blind constants, the shipped pick was
an isolated defect and not a class, and this row's framing is wrong.

## P1 — the Blackhole step ratio

Blackhole here is qb2 card 1, an 11x10 = **110-core** grid, against whglx's 8x9 = 72.

The rule at the shipped diffusion shape (512 tokens, padded 512, 16 heads, `work = 16`) picks the
smallest `q_chunk` dividing 512 whose `16 * 512 / q_chunk` fits the grid:

* 72 cores: `q_chunk = 128` -> 64 units. **Measured 1.3117x on the op, 1.01831x on the step.**
* 110 cores: `16 * 512 / 128 = 64 <= 110` and `16 * 512 / 64 = 128 > 110`, so the rule picks
  **128 there too**. The pick does not move between the two architectures at this shape.

That is the interesting part and it sets the prediction. The shipped 256 is *worse* on Blackhole in
occupancy terms (32 of 110 cores busy, 29 %, against 32 of 72, 44 %), while the rule's pick is the
same chunk on both. So if occupancy is the whole mechanism, **the Blackhole win should be larger
than the Wormhole one, not smaller.**

**Predicted BH step ratio: 1.020x - 1.045x**, centre **1.028x**. I am explicitly predicting the
parent's 1.00475x fold PROJECTION is an under-estimate, because it transported a Wormhole *step*
ratio onto a Blackhole step wall instead of re-deriving the occupancy.

**Falsifier for P1:** a measured Blackhole step ratio below 1.015x means the win is not occupancy
alone — most likely the K/V re-read a narrower chunk pays is worth more on the part with more L1
per core, which would mean the rule needs a second term and not just a grid term.

## P2 — across models, not just sizes

Six call sites pass `work` today (`tenstorrent.py`, `esmc.py`, `esmfold2.py`, `saprot.py`).
Predicted: the rule changes the pick only where `work * padded / 256 < n_cores`, i.e. on the
*small* work-unit shapes, and leaves the atom SDPA (560 heads) and every large-batch PLM shape
exactly where they are. **Predicted: no model loses on any shipped shape.** I expect the pick to
move on Boltz-2/BoltzGen's diffusion token SDPA and on at least one PLM shape at short sequence.

**Falsifier for P2, which is the brief's own:** a loss on ANY shape or ANY model means the grid
rule is not the whole story. The parent measured worst case 1.0000x over ten shapes; anything under
1.0 is new information and gets published as such, not smoothed.

## P3 — parity

Bit-exact at every rung, `torch.equal`, max abs 0.0, re-run on this build rather than inherited.
`q_chunk` partitions independent query rows and the reduction order lives in `k_chunk`, which this
row does not touch. **If any rung is not bit-exact, the mechanism is not what the parent said it
was** and the lever stops being free.
