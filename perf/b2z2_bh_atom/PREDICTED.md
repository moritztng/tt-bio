# b2z2-bh-stack-atom — PREDICTION, written before the first fold of this row

Written 2026-09-12, before any device was opened by this row. Not edited afterwards. ARCH: every
number below is a Blackhole prediction (qb2 card 1, one processor of a p300c, 11x10 grid, ttnn
0.68.0), except where it names Wormhole as the source it extrapolates from.

## 1. The gate's arithmetic on the 11x10 grid, done on paper first

`_atom_branch_memory_config` admits L1 when

    live = B * K * D * (W + 4 * ATOM_DIM) * elem  <=  SHARE * _l1_bank_bytes() * gx * gy

with `W = ATOM_WINDOW = 32`, `ATOM_DIM = 128`, `SHARE = 0.5`, `B = 1`, `D = 128` (the atom single
channel count) and `elem = 2` for bf16. So `live = K * 128 * 544 * 2 = 139264 * K` bytes.

`_l1_bank_bytes()` reads the L1 allocator, documented in `tenstorrent.py` as **1461760 B per bank on
Blackhole**. On the main grid `gx * gy = 11 * 10 = 110`, so the budget is

    0.5 * 1461760 * 110 = 80,396,800 B = 80.40 MB

against Wormhole's `8 * 9 = 72` banks, roughly **49 MB**. **The Blackhole budget is ~1.64x the
Wormhole one**, and the live set does not depend on the grid at all. So the gate gets *looser* on
Blackhole, not tighter, which is the opposite of the way this brief's risk is usually framed.

`K` is the window count, `atom_axis / 32` with `atom_axis = ceil(atoms / 448) * 448`. At 512 aa the
campaign's own number is 140 windows / 4480 atoms, i.e. 8.75 atoms per residue on this fixture, and
the `cdk2x2` ladder is the same chimera repeated, so K should scale very close to linearly.

| size | predicted K | predicted live | budget | predicted gate |
|---|---|---|---|---|
| 256 aa | ~70 | ~9.75 MB | 80.40 MB | **L1** |
| 512 aa | 140 | **19.50 MB** | 80.40 MB | **L1** |
| 768 aa | ~210 | ~29.24 MB | 80.40 MB | **L1** |
| 1024 aa | ~280 | ~38.99 MB | 80.40 MB | **L1** |

**Prediction: `ATOM_L1_STATS` reads `dram: 0` at all four sizes.** The gate does not decline anywhere
on this ladder; on these shapes it would not decline until roughly 2900 aa.

**So the pre-registered falsifier — "`TT_BIO_ATOM_L1` declines on the 11x10 grid at 512 aa" — is
predicted NOT to fire, and I give it under 10 %.** If it does fire it is a first-class result and it
gets the headline.

**But the ladder's real risk is not the gate, it is the allocator, and the gate is what fails to
protect against it.** The budget counts only the atom branch's own four tensors. It does not count
the program CBs, the sampler's other live activations, or the fact that `ttnn.L1_MEMORY_CONFIG` is
interleaved across every bank, so a 39 MB live set at 1024 aa is competing with everything else in
`Diffusion.__call__` for the other half of L1.
`of3-1024aa-oom-allocation-count-not-size` is the live memory here: allocation COUNT at the failure
point, not the size of the one that failed, is what decides this class. **I put OOM at 256 and 512 aa
at under 5 %, at 768 aa at ~25 %, and at 1024 aa at ~45 %.** An OOM at 768 or 1024 aa is a NO-GO for a
default flip and I will say so plainly; the cheap remedies, in order, are lowering
`TT_BIO_ATOM_L1_SHARE` and making the gate count the live set it actually creates rather than the
four tensors it names.

`tt-bio-tuned-at-512-l1-gates-go-dark-above-640aa` predicts the *other* failure shape at these sizes:
three unrelated 512-tuned gates go dark between 640 and 1024 aa with no error and no log. If 768 or
1024 aa is slow rather than broken, that is those gates, not this lever, and I will not attribute it
to this lever without a counter.

## 2. Step and fold ratios

`TT_BIO_ATOM_L1` measured **1.08461x on the grabbed-step replay on Wormhole**. Two corrections apply
before that becomes a Blackhole fold number, and they push in the same direction:

1. `b2z2-sampler-union-wh` measured its own grabbed step over-predicting the in-fold sampler by
   **4.80 %**, and `rfd3-isolated-screen-underprices-residency-lever`'s mirror image applies here: a
   replay harness truncates the stacks that would compete for L1. **1.08461x is an upper bound.**
2. Blackhole has 1.64x the L1 budget and 1.53x the banks, so the residency is if anything easier to
   hold; but its DRAM roof is the thing the lever is buying against, and this row has no reason to
   expect the ratio to grow.

The two step levers that have already crossed WH -> BH transferred at 0.991x and 0.9999x of their
Wormhole ratio, so I take transfer near 1.0 as the base case.

| quantity | predicted range | point |
|---|---|---|
| `TT_BIO_ATOM_L1`, BH replayed step | 1.055 – 1.090x | **1.075x** |
| `TT_BIO_ATOM_L1`, BH in-fold sampler wall | 1.015 – 1.050x | **1.030x** |
| `TT_BIO_ATOM_L1`, BH fold | 1.004 – 1.014x | **1.008x** |

**The fold point estimate is below the cell's 1.01162x A/A floor.** So I pre-register that
`TT_BIO_ATOM_L1` alone is predicted to be **not measurable on the fold** on this box, and the sampler
wall is the instrument that has to carry it. A fold-level null for this lever alone is a floor
result, not a refutation, and I will not report it as one.

## 3. The corrected stack

The parent's five-lever stack is **1.12862x**, and it contains `TT_BIO_ATOM_KEY_WINDOW` at
**1.02744x** on the BH fold. The correct implementation of that same transform,
`TT_BIO_ATOM_SHIFT_GATHER`, measures **1.01985x** on the BH fold. Swapping one for the other should
cost `1.02744 / 1.01985 = 1.00744`, so:

| arm | predicted range | point |
|---|---|---|
| corrected five-lever stack (SHIFT_GATHER for KEY_WINDOW) | 1.112 – 1.128x | **1.1204x** |
| + `TT_BIO_ATOM_L1` | 1.118 – 1.140x | **1.1294x** |
| + `TT_BIO_ATOM_KV_PREPROJ` (marginal, on the shift construction) | 1.120 – 1.145x | **1.1322x** |

`TT_BIO_ATOM_KV_PREPROJ` is the marginal figure only: its 1.01566x headline was measured on top of
the key window, and `b2z2-sampler-union-wh` re-measured it on the shift construction at **1.00947x on
the step**, which Amdahl's ~26 % sampler share turns into about **1.0025x on the fold** — under the
floor by a factor of four. I predict it is **not separately measurable on this fold** and that the
honest report is a sampler-wall number plus a stack that contains it.

Sub-additivity: the parent measured its own stack +0.24 % against the product of singles, and
`b2z2-sampler-union-wh` measured a 2.169 % discount on a three-lever atom union. I predict the
corrected stack lands **0 – 3 % below the product of its parts** and I will report the discount.

## 4. Parity

The three arms I am adding or swapping in are all bit-exact, and two of them were proved bit-exact
against the *definition* and not against each other. So:

**Prediction: the corrected stack's parity is unchanged from the parent's, 0.49519 Å at 512 aa and
0.42995 Å at 298 aa, to within the scorer's own noise — which the parent measured as exactly zero,
because its A/A repeat was byte-identical.** I predict the corrected stack's CIF differs from the
parent stack's CIF (the key window and the shift gather disagree by 3.828 in windows that 512 aa
slices away, so at 512 aa I expect them to agree; at 298 aa the atom bucket tail is different and I
am genuinely unsure). **If the stack's RMSD moves when I add a bit-exact lever, the lever's gate is
wrong and that is the finding**, and it outranks every ratio in this document.

## 5. What would make me report a NO-GO

* `ATOM_L1_STATS` reads `dram > 0` at 512 aa — the wave's largest bit-exact lever does not exist on
  the cell. Predicted under 10 %.
* An OOM at any ladder size — no default flip for `TT_BIO_ATOM_L1`, stated plainly, with the size.
* The corrected stack's worst per-pseudo-domain reading crossing 0.60 Å.
* The corrected stack measuring at or below the A/A floor above the parent's swap-adjusted 1.1204x —
  i.e. the two atom levers buying nothing on the cell.
