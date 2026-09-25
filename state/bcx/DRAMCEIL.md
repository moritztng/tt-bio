# Where a BindCraft 2 gradient step stops fitting on one Blackhole

bcx-armtree, 2026-09-25, qb2 card 3 (p300c, `0000:04:00.0`), 34.226 GB of DRAM.

## The short answer

**Every binder length BindCraft 2 can draw fits on one card, on the fixed tree.** BC2 draws 60 to
180 against the 115-residue hPDL1 target; the largest draw, 180, runs six consecutive gradient
steps with the resident DRAM line perfectly flat. Nothing in the 60-180 range is lost.

On a tape whose storage groups are held STRONGLY, the largest draw does not survive one step. The
weak-group fix is what moved the ceiling, and it moved it off the top of BindCraft 2's own range.

## What the allocator actually sees

Not the drawn length. Three paddings sit between the draw and a ttnn buffer:

1. `pad_design_chains` (`bindcraft/af2.py:58`) rounds the binder chain up to `length_bucket_size`, 32;
2. `_predict_complex` (`af2.py:298`) rounds binder-plus-target up to 32 again;
3. `splice._pad_inputs` rounds the token axis up to a multiple of 32 for tt-bio. At bucket 32 this
   one is already satisfied by the step above it.

So the 121-value draw range collapses to **five** padded sizes. From `perf/bcx_armtree/padmap.py`,
which runs BindCraft 2's own padding functions on every length 60..180 (CPU, no device):

| padded tokens | draws that produce it |
|---|---|
| 192 | 60..64 (5 draws) |
| 224 | 65..96 (32) |
| 256 | 97..128 (32) |
| 288 | 129..160 (32) |
| 320 | 161..180 (20) |

The device-side number is read off the real call rather than computed: `ceil.py` wraps
`splice._pad_inputs` and records its `n32`. At L=180 it reports `device_n = 307`,
`device_n32 = 320`, one distinct size across every call in the step.

**This is why `l166` died and `l142`/`l146` did not.** 166 pads to 320; 142 and 146 both pad to
288. The draw was the discriminator because of which side of the 160/161 bucket edge it fell on,
and `l166` shares its padded size with BindCraft 2's maximum draw of 180.

## The fixed tree: `a1797d5dd`, 2 weakrefs in `autograd.py`, clean worktree

`ceil.py --binder-length 180 --steps 6`, screen stage, AICLK median **1350 MHz** (min 1312, max
1350, 1307 samples taken during the timed steps), loadavg 10.4-12.8 with three other campaigns on
the box.

| step | seconds | DRAM before | forward seam | after |
|---|---|---|---|---|
| 1 | 76.4 | 0.2191 | 1.4634 | 0.5823 |
| 2 | 57.5 | 0.5823 | 1.4668 | 0.5823 |
| 3 | 59.9 | 0.5823 | 1.4668 | 0.5823 |
| 4 | 57.7 | 0.5823 | 1.4668 | 0.5823 |
| 5 | 57.2 | 0.5823 | 1.4668 | 0.5823 |
| 6 | 58.2 | 0.5823 | 1.4668 | 0.5823 |

Resident rise over six steps: **0.0000 GB**. Step 4 is the step `bcx-mutate`'s seed-2 arm never
finished; here it is unremarkable. Artifact `perf/bcx_armtree/ceil_fixed_L180.json` on
`wk/bcx-armtree`.

## The leaky control: the same commit, one file inverted

`_share`/`_members` reverted to strong references in `/home/ttuser/bcx_armtree_leaky`
(`weakrefs_in_autograd = 0`, `dirty_autograd = true`, 11 insertions and 9 deletions in one file,
nothing else different from `a1797d5dd`).

| draw | padded | outcome |
|---|---|---|
| 180 | 320 | OOM in **step 1, backward**. 29.72 GB allocated, 2.16 GB free, refused a 524,288,000 B buffer across 8 banks |
| 160 | 288 | OOM in **step 1, backward**. 30.34 GB allocated, 1.53 GB free, refused a 191,102,976 B buffer |

The 320 failure carries the same refusal size, 524,288,000 B across 8 banks, that killed
`bcx-mutate`'s `pdl1_denovo_l166_80a6a681416874e6`.

Both arms reached the backward before dying (`failed_phase = backward`, two taped forwards
completed), so neither is a forward OOM.

**This control is harsher than the trees the live arms run, and the bound is stated rather than
smoothed over.** `bcx-mutate`'s siblings survive many steps at 288 while this control does not
survive one, so `a1797d5dd` with strong groups is not a stand-in for the older trees in flight;
something else on the merged tree depends on the weak lifetime. What it does establish: on this
tree the weak storage group is load-bearing at both 288 and 320, not a tidy-up. Placing the first
failing draw on the actual in-flight trees (`1127f9ea8` and the `bcx-mutate` tip) needs one probe
per tree and is the obvious next chunk.

## Peak allocation

The two seam reads bracket the forward and the backward, so **1.4668 GB is a lower bound** on the
instantaneous peak, not the peak: a transient inside a block is invisible to them. `bcx-large`
measured the instantaneous figure with per-node sampling and got 16.698 GB at n=352 of 34.226 GB,
so the real in-step peak at 320 is nearer that order than to 1.47. A node-sampled run at this
configuration is in flight; its artifact is `perf/bcx_armtree/ceil_fixed_L180_nodepeak.json`.

## What fraction of the draw range we lose

**None of it, on the fixed tree.** 0 of BindCraft 2's 121 drawable binder lengths (60..180) fail,
and the top bucket runs six steps with a flat resident line. On a strong-group tape at least the
top two buckets go, 52 of 121 draws (43 %) at minimum, and on the merged tree all of them.

## For CMP

The capability claim that reads "~954 padded residues off BindCraft 2's own memory formula" has
never been checked against silicon and this row does not confirm it. What silicon says is narrower
and stronger: **at the sizes BindCraft 2 actually asks for, every one of them fits on a single
Blackhole card, measured, six steps deep, at 1350 MHz.** The largest padded size the campaign can
produce is 320 tokens, not 954.
