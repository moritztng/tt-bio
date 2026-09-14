# The accuracy budget, anchored to a non-Tenstorrent reference

The K10 campaign rations a 0.60 Å budget of which the b2z2 stack has spent 0.49519 Å at 512 aa.
Every one of those Ångström was measured against another Tenstorrent fold, so the budget was
drift-from-last-shipped, not error-against-truth. This measures the missing side: the same two
fixtures, the same protocol, folded by the upstream package.

## The reference

| | |
|---|---|
| stack | `boltz==2.2.1` from PyPI, not `boltz[cuda]`, `--no_kernels` passed as well |
| precision | fp32. Lightning `precision=32`, overriding the `bf16-mixed` boltz hard-codes for every boltz2 run (`boltz/main.py:1262`) |
| TF32 | off, matmul and cuDNN, set before every fold |
| hardware | 1x RTX 6000 Ada, driver 595.58.03, torch 2.14.0+cu130, rented for this measurement |
| protocol | 3 recycles, 200 sampling steps, 1 diffusion sample, seeds 0-3, `--use_potentials` off |
| input | `perf/size512/fixtures/cdk2x2_{298,512}.{yaml,a3m}`, the committed 35-row a3m, unmodified except for the `msa:` key |

The CIFs are in `gpucif/`, the per-fold record in `out/runs_{298,512}.json`.

Two things make this a reference rather than a second approximation. It is a different
implementation on different hardware, so it shares no arithmetic with the port. And it runs fp32:
a bf16 reference would have an 8-bit mantissa, which is the same precision class as the thing it
is supposed to anchor. The `gpurefbf16` arm below measures what that choice was worth, and it is
not small.

Atom identity was checked rather than assumed: the `(chain, seq id, atom name, residue)` key list
of the upstream CIF equals the TT CIF's at both sizes, so the two stacks are compared atom for
atom with no reindex.

Scored with `perf/b2z2_fusebias/score.py`, unmodified, the instrument the rest of this campaign
uses. The reference is the baseline arm, so what the scorer calls "the lever" is main's deviation
from upstream.

## The seed floor, re-measured on the reference stack

The 1.84 Å constant the whole accuracy policy is scaled against came from one seed pair on one
stack. Six pairs per arm, worst per-pseudo-domain all-atom Kabsch RMSD:

| | 512 aa | 298 aa |
|---|---|---|
| upstream reference, seeds 0-3 | **1.66454 Å** mean, 0.94540 - 2.29123 | **0.80128 Å** mean, 0.74099 - 0.84656 |
| TT main, seeds 0-3 | 1.23804 Å mean, 1.06815 - 1.39420 | 1.01758 Å mean, 0.79244 - 1.25086 |
| quoted constant | 1.84002 Å | 0.82073 Å |

The quoted constants survive: 1.84002 Å sits inside the reference stack's measured 0.945 - 2.291 Å
range at 512 aa, and 0.82073 Å sits at the top of its 0.741 - 0.847 Å range at 298 aa. What changes
is that the floor is now a distribution with a known width instead of a single number, and the
width at 512 aa is large: one seed pair of the reference differs from another by 2.29 Å.

## Main's deviation from the reference

Worst per-pseudo-domain all-atom, main against the reference at the same nominal seed:

| | 512 aa | 298 aa |
|---|---|---|
| main vs reference | **1.42726 Å** mean, 1.05383 - 2.18080 | **0.90077 Å** mean, 0.83382 - 0.96860 |
| reference's own seed floor | 1.66454 Å mean | 0.80128 Å mean |
| ratio | **0.86x** | **1.12x** |
| 0.60 Å bar | outside | outside |
| per-pseudo-domain CA | copy 1 0.99638 Å mean (max 1.80442), copy 2 0.79720 Å mean (max 0.97753) | 0.55038 Å mean (0.40324 - 0.64336) |
| whole-structure all-atom | 8.28422 Å mean, 1.92709 - 11.94438 (the hinge) | same as per-domain, one domain |
| whole-structure CA | 8.09572 Å mean, 1.78002 - 11.69181 (the hinge) | 0.55038 Å mean |
| CA-lDDT, main vs reference | 0.91056 worst | 0.97673 worst |

At 512 aa main is closer to the upstream reference than the upstream reference is to itself at a
different seed. At 298 aa it is 12 % further. Neither number is inside the 0.60 Å bar, and neither
can be: the two stacks do not share the sampler's noise, so a cross-stack RMSD carries the full
sampling spread whatever the arithmetic does. That is the reason the whole-structure column swings
from 1.9 to 11.9 Å at 512 aa while the per-domain column barely moves - the chimeric fixture's free
hinge, exactly as `cdk2x2-chimeric-fixture-cannot-score-non-bit-exact-parity` describes.

## Against the experimental structure

1HCL is CDK2 solved by crystallography. It is the one thing in this measurement that no sampler
argument can reach, and `cdk2x2_512` is CDK2 followed by its own residues 1-214, so both
pseudo-domains have a native answer. Four seeds per arm, CA-lDDT and CA RMSD against 1HCL:

| | reference (fp32) | TT main | separated? |
|---|---|---|---|
| 298 aa, CA-lDDT | 0.97762 mean, 0.97045 - 0.98134 | 0.96741 mean, 0.95950 - 0.97806 | no, ranges overlap |
| 298 aa, CA RMSD vs 1HCL | 0.68098 Å mean | 0.76680 Å mean | no |
| 512 aa copy 1, CA-lDDT | 0.95601 mean, 0.93254 - 0.96546 | 0.94036 mean, 0.93585 - 0.94797 | no |
| 512 aa copy 2, CA-lDDT | 0.94264 mean, 0.93236 - 0.94940 | 0.91860 mean, 0.91317 - 0.92654 | **yes** |
| 512 aa copy 2, CA RMSD | 0.97354 Å mean, 0.87185 - 1.05967 | 1.15216 Å mean, 1.08673 - 1.23405 | **yes** |

Copy 2 is the one place the two stacks separate cleanly: four seeds against four, TT main's best
CA-lDDT (0.92654) is below the reference's worst (0.93236), and the same holds on CA RMSD. Under a
permutation test complete separation of two groups of four has probability 2/70 = 2.9 %, so this is
a real signal rather than a wide-floor artifact, but n=4 is n=4.

## Two controls, and they carry the whole argument

**The b2z2 stack did not spend the budget.** `ttbase` is the same TT tree with
`TT_BIO_DEVICE_CONDITIONING=0`, four seeds, already committed under `perf/b2z2_cond/cif/`. If the
0.49519 Å the stack spent were an accuracy loss, `ttbase` would be closer to the reference and
closer to 1HCL than `ttmain`. It is neither:

| | 512 aa | 298 aa |
|---|---|---|
| vs reference, worst per-pseudo-domain | ttbase 1.54370 Å, ttmain 1.42726 Å | ttbase 0.88286 Å, ttmain 0.90077 Å |
| CA-lDDT vs 1HCL, copy 2 / whole | ttbase 0.91573, ttmain **0.91860** | ttbase 0.96742, ttmain **0.96741** |

At 298 aa the two TT arms score identically against the experimental structure to four decimals.
At 512 aa main is slightly better than pre-lever on both readings. The levers moved coordinates
inside the sampler's own distribution; they did not move accuracy.

**An unpaired comparison costs ~1 Å on this fixture no matter what the arithmetic does.**
`gpurefshared` is the reference stack folding itself: same package, same fp32, same GPU, same seed
number, the only change being that the sampler's noise is drawn on the CPU and reseeded at the
sampler entry instead of running off the CUDA stream. Nothing arithmetic differs. It lands
**0.92565 Å** from `gpuref` at 298 aa and **1.82527 Å** at 512 aa.

Those are the numbers to hold against main's 0.90077 Å and 1.42726 Å. **Main is closer to the
upstream reference than the upstream reference is to itself with a different noise realisation**,
at both sizes. The 1.82527 Å also reproduces the 1.84 Å constant the accuracy policy is scaled
against, from a completely different direction.

**bf16 is not where the difference lives.** `gpurefbf16` is the reference stack at boltz's shipped
`bf16-mixed` instead of fp32, same seed, same CUDA noise stream, so the comparison is paired and
arithmetic-only. It moves the structure **0.01993 Å** at 298 aa and **0.01390 Å** at 512 aa, and
CA-lDDT against 1HCL is unchanged at four decimals (298 aa: 0.97767 vs 0.97762; 512 aa copy 2:
0.94269 vs 0.94264). Its seed floor matches the fp32 floor to within 0.002 Å (1.66591 vs 1.66454 Å).
Narrowing Boltz-2's activations to bf16 costs nothing measurable on this fixture.

That also sets the scale for reading every other number here: an arithmetic-only change on this
model reads 0.01-0.02 Å. Anything reading 0.9-1.8 Å is the sampler, not the arithmetic.

## What the budget actually is

The 0.60 Å bar is a **paired, same-stack, same-seed** bar. A lever is folded against the
unmodified tree at the same seed on the same card, so the sampler's noise cancels and what is left
is the lever. That is why a lever reads 0.49519 Å while two innocent runs of the same code read
1.84 Å.

It cannot be re-pointed at a cross-stack reference, because there is no shared noise to cancel:
the control above shows the same code compared to itself unpaired reads 0.92565 / 1.82527 Å. Any
"main vs upstream" RMSD is a floor of about 1 Å at 298 aa and 1.8 Å at 512 aa before arithmetic is
allowed to contribute anything.

So the answer to the question Phase 4 needs:

- **None of the 0.60 Å budget has been spent in accuracy-against-truth terms.** The pre-lever and
  post-lever TT arms score the same CA-lDDT against 1HCL (0.96742 vs 0.96741 at 298 aa; 0.91573 vs
  0.91860 at 512 aa, main ahead). The full 0.60 Å remains allocatable on the bar's own terms, and
  the bar's own terms are the correct ones.
- **The bar is not the binding constraint anyway.** The port carries a standing CA-lDDT offset
  against upstream that predates the campaign: -0.0102 at 298 aa, -0.0156 on copy 1 and -0.0240 on
  copy 2 at 512 aa, with copy 2 fully rank-separated over four seeds per arm. No lever in this
  campaign created it and no lever in this campaign will be visible against it, because a lever
  costs 0.5 Å of coordinates and 0.000 of lDDT while the offset costs 0.024 of lDDT.
- **Phase 2's accuracy-spending levers are not dead on arrival.** The opposite: the evidence says
  RMSD-against-last-shipped over-reads what a lever costs, and the reading with teeth is CA-lDDT
  against the experimental structure, which no lever measured so far has moved.

## What is owed

A TT fold with `TT_BIO_SHARED_DRAW_SEED=0` at both sizes, scored against `gpurefshared`. That is
the only comparison that would give an arithmetic-only TT-vs-upstream number, the way
`gpurefbf16` gives one for the precision question. The reference half exists and is committed
(`gpucif/{298,512}_gpurefshared-s0`); the TT half needs a card, which this task did not hold.
It would turn the -0.024 CA-lDDT offset from "measured but unexplained" into a located one.
