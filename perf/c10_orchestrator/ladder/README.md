# The C10 ladder

Baseline **14.8813 s at 512 aa** and 9.6801 s at 298 aa, pinned 1350 MHz, 16 folds per size
(`c10-bare-baseline` `bc66f7d6d`). Target 10.0 s, so the gap is **4.881 s**.

Every row carries an evidence class, because a number measured on Blackhole at a recorded clock
and one transferred from Wormhole at an unrecorded clock are not the same kind of thing.

| candidate | worth | evidence | status |
|---|---|---|---|
| ttnn trace of the diffusion loop | **0.00 s** — sized at 3.02 s, measured −0.0214 s | Blackhole, pinned 1350 MHz | **concluded NO-GO** |
| the whole clock-immune term | unknown, 3.98 s in play | not measured anywhere | **unowned** |
| per-class grid sizing | withdrawn | transferred across architectures | queued |
| `TT_BIO_SDPA_GRID_Q_CHUNK` sign | −0.46 to +0.21 s | not measured anywhere | queued |
| `TT_BIO_HEAD_PAD_TAIL` | 0.21 s / 284 Mcyc | Wormhole, no clock | not rowed |
| `TT_BIO_DIT_FUSED_QKV` | 0.12 s / 162 Mcyc | Wormhole, no clock | not rowed, **excluded** |
| the byte axis, everything remaining | 1.487x ceiling | Blackhole, no clock | reopened: it is the only known lever against `F` |
| size-independent share of the work term | 3,301 to 8,220 Mcyc / 2.45 to 6.09 s | bounded from two measured sizes | `c10-size-scaling`, now carrying counter-evidence |
| fuse the arithmetic-free elementwise traffic | **0.69 s / 932 Mcyc** | derived from two roofs plus a measured fusion return | **unowned** |

**Priced total 0.90 s, so the fold would read 13.98 s** — 18.4 % of the gap. It was 4.16 s reading 10.72 s until
`c10-trace-lever` reported: 3.02 s of that total was the diffusion trace, and the trace returns
nothing. It is back above half a second because
[`../arithmetic_free_traffic/`](../arithmetic_free_traffic/) found that three op classes move 30.8 %
of the fold's bytes while computing nothing. The naive sum is 1.02 s, but `TT_BIO_DIT_FUSED_QKV` and `TT_BIO_HEAD_PAD_TAIL` are jointly
0.713 A against a 0.60 A bar and cannot both ship, so the cheaper one drops out.

## Read this before quoting the total

**Nothing in this table has been measured as a fold-level win.** Cycles saved to date: zero.
Accuracy spent to date: none. The total is an inventory of what is worth trying, not a forecast.

- Two priced rows are *derived* — arithmetic over measured artifacts rather than measured levers.
- Two are Wormhole numbers that have never run on Blackhole.
- Perturbations stack strongly sub-additively on this fixture, so a stack has to be measured as a
  stack. Summing individual readings is exactly the error the campaign forbids.

**At full value the priced rows land at 13.98 s**, still 3.98 s above the target. That is the
honest state of the campaign: the largest candidate it ever had measured to zero, and what has
replaced it is a third of a well-understood fusion rather than anything that closes the gap.

The live items all point at the same two terms of `T = F + W/f` that `c10-fixed-cost` separated:

- **The size-independent share of the work term, 3,301 to 8,220 Mcycles.** The clock-scaled work
  grows at N^0.63, slower than the target does, which bounds a term that does not grow with the
  target at all. That is between half of the campaign's whole deletion target and more than all of
  it, and nothing has ever attacked it. It could still be 298 aa under-filling the grid;
  `c10-size-scaling` settles that at 640 and 768 aa. See [`../size_scaling/`](../size_scaling/).
- **The byte axis, against `F` rather than against kernel time.** `F` is 3.98 s, 27 % of the fold,
  and two independent lines say it is not host overhead: the trace null, and `F` scaling at
  N^1.32 ± 0.07. A term that grows at N^1.3 and survives dispatch removal looks like DRAM-bound
  device time, and bytes are the only lever against that. This now has a target rather than a
  ceiling: `multiply_`, `layer_norm` and `add_` move 0.881 TB and compute nothing.
`c10-fixed-cost` is measuring the term; nothing is attacking the remainder.

    python3 ladder.py                      # prints ladder.json
    python3 -m pytest test_ladder.py -q    # 9 controls
