# BindCraft 2 on a Wormhole Galaxy chip

The first BindCraft 2 gradient rounds to run on Wormhole. Dev Galaxy `.107` (`UF-EV-A4-GWH01`,
Osaka dev box, not production), **chip 30** = sysfs node 6, PCI `0000:c7:00.0`, `Arch.WORMHOLE_B0`,
compute grid **8x9**, so `tt_bio.tenstorrent._IS_SMALL_GRID` is the path in force. tt-bio
`2d174fd5b`, ttnn 0.68.0 (`cp312` wheel), BindCraft 2 `7a2dfdb`, AF2 parameters 2022-12-06.
The chip was borrowed through tt-bio's own lease in the agent's lease dir. Note what that costs:
stopping and restarting the box's japanfold agent takes **all ~32 of its chips** out of
`/v1/cluster` for the restart window, not just the one being borrowed. Harmless here because dev
had 0 jobs running, which was checked before every handover, and not harmless otherwise.

`out/*.json` are the harness's own event logs, one per arm. Each carries its AICLK samples,
taken at 1 Hz **during** the run from the chip's sysfs; every reading below is at a median of
**1000 MHz**, which is the Wormhole ceiling.

| arm | tokens | configuration | warm round | device | host |
|---|---|---|---|---|---|
| `comp11_b146` | 288 | composed, extra-MSA and template on card | **15.72 s** | 12.71 s | 3.01 s |
| `comp_b146` | 288 | composed, extra-MSA and template in JAX | 26.62 s | 11.77 s | 14.85 s |
| `grad_b60` | 175 | `exact=1` tape (not what tt-bio ships) | 347.2 s (cold) | 324.8 s | |

The Blackhole peer is the same arm at the same token axis: `perf/bcx_p10_duotraj/duo_round.py`
hardcodes `extra_msa=True, template=True, exact=False` and its `arm.sh` exports no `TT_BIO_*`
lever because those are main's defaults.

| | Blackhole | Wormhole | ratio |
|---|---|---|---|
| N=1, 288 tokens, seed 100 | 7.204 s @ 1350 MHz | 15.72 s @ 1000 MHz | **2.18x** |
| default to default | 5.899 s (N=3 auto) | 13.21 s (N=2 auto) | **2.24x** |

Folding measures Wormhole/Blackhole at 2.27-2.38x at matched input, so BindCraft 2 costs about
what folding costs on this board and not less. 1.35x of the gap is the clock: 1000 MHz is this
part's architectural maximum, measured as max-1000 over 2946 samples, not a clamped sitting, and
the Blackhole rule of thumb that a reading under 1200 MHz is an artifact does not carry here.

An earlier version of this file compared against 11.206 s and reported 1.40x. That figure is
`bcx-p10-hostcut` from 09-26 and main has moved 236 commits of levers since; it is kept here only
as the anchor that was corrected.

Every fused kernel BindCraft 2 ships serves on the 8x9 grid, with nothing declined for a
Wormhole reason: `fused_hifi` 1248 served / 0 declined, `triatt_bw` 416 / 0, `rne_add` 5424 / 48
(the 48 are a shape mismatch that is not board-specific). `mm_layout`'s declines are all shape
reasons it gives on Blackhole too.

Two traps cost a pass each and are worth knowing before you re-run this:

- `perf/bcx_round/run_round.py --exact` defaults to 1 and its help calls that origin/main's
  default. It is not: `bindcraft2.predictor` ships `exact=False`, and the exact tape is 24.87x
  the round. A `--exact 1` reading is not a Wormhole regression, it is the slow tape on any board.
- `--exact 0` alone aborts at the first round boundary with `lever went inert mid-arm`, because
  `exact=False` makes `bindcraft2.fast_round` arm six levers the harness was not told to expect.
  Ask for them: `--triatt-hifi 1 --rne-kernel 1 --triatt-bw 1` with `TT_BIO_MM_LAYOUT=1
  TT_BIO_TAPED_CHANNEL_MOVE=1 TT_BIO_WIDEN_ADD=1`.

## The campaign, and the gradient

A real `examples/pdl1.json` campaign ran to its own stop condition in 49 min 6 s and accepted
**0 designs of 2 trajectories**, both rejected at `harden` on `[i_pTM, pLDDT]`. That is an
ordinary Blackhole outcome, not a Wormhole fault, and three of BCX's own instruments say so:
main's README puts Blackhole at 7 accepted per 31 trajectories, at which P(0 in 2) = 0.60;
`harden` is the modal terminal stage on Blackhole too; and the round-count-matched step from
`stage_window.py` reads -0.19 and -0.40 here against Blackhole's eight draws spanning +0.26 to
-0.37 and BindCraft 2's own JAX reference at -0.64. The printed anneal->harden step is partly an
artifact of a max over 45 rounds against a max over 5, which is why the matched step is the one
to read. Pooled jitter 0.090 and 0.060 sits inside a Blackhole distribution whose rejected and
passed sets fully overlap.

The gradient is correct on this board. `afgrad.py vjp --n 288 --blocks 4,5,6,7 --controls-all`
against a float64 reference on the same bf16-rounded inputs, `out/grade/vjp_n288_wh_chars.json`:

| block | device dz | torch bf16 | ratio | cos | permuted cos | zero seed |
|---|---|---|---|---|---|---|
| evo0 | 0.0188656 | 0.0171981 | 1.10x | 0.999822 | -0.000843 | 0.0 |
| evo1 | 0.0163565 | 0.0157703 | 1.04x | 0.999866 | 0.000114 | 0.0 |
| evo2 | 0.0194055 | 0.0183585 | 1.06x | 0.999812 | 0.000358 | 0.0 |
| evo3 | 0.0161863 | 0.0164162 | 0.99x | 0.999869 | 0.000037 | 0.0 |

The bar is torch's own bf16 error against the same float64, not zero, because the device computes
in bf16. The controls are what make it a result: a permuted cotangent collapses `cos` below
0.001, and a zero seed returns exactly 0.0.

`out/grade/vjp_n288_wh_hifi.json` is kept for honesty rather than for its conclusion. That run,
scoped to `--blocks 4,5` with no controls, returned evo1 dz 5.21e+32 with `norm_ratio` Infinity
while its own evo0 was fine and bit-identical to the run above, and while that same evo1's
forward was fine. A re-run of those exact arguments then died before scoring anything, with
`Signal: Bus error (7) / Non-existent physical address (2)` inside `ttnn::layer_norm`.

So that configuration has failed twice out of two, once silently and once loudly, and both are
the signature of a bad memory access rather than of arithmetic. The four-block grade above is
still the answer to "is the gradient correct on this board" -- it is -- but there is an
intermittent memory fault reachable from the AF2 backward here, and the silent form of it is a
wrong gradient that does not crash. Root-cause it before treating Wormhole as ready for
BindCraft 2. The product path is not implicated: 210 campaign rounds ran clean, and the fault
has only appeared under this file's teacher-forcing pattern, which re-enters individual blocks
out of sequence with synthetic cotangents and is not what `run_campaign` does.

Do not quote the 1e32 as a Wormhole performance or accuracy result.
