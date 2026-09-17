# The 15.031 s floor and the 2.309 s prize, checked against the measured terms

Both numbers have been quoted all campaign and both live in one committed artifact,
`perf/roof_true/true_floor_512_qb2c2.json`. `c10-roofline-reset` could not revalidate them because
its byte counter failed a known-answer control. `c10-fixed-cost` has since measured the two things
that artifact models, so they can be compared directly.

## The prize was the clock. It is retired.

The artifact's own identity is `prize_s = cell_of_record_s − floor_s`, that is 17.34 − 15.031 =
2.309 s, and it holds exactly in the file. But 17.34 s is what `fold_s = 2.901 + 15355/AICLK`
returns at about 1063 MHz — it was never a slower code path, it was a throttled measurement.

The same fixture at a pinned, during-sampled 1350 MHz reads **14.846 s**, which is **0.185 s below
the floor the prize was measured against**. Whatever the true floor is, the fold is not 2.309 s
above it. The 2.309 s of "remaining prize" that sized this campaign's ambition was the clock.

## The floor's two halves land close to the two measured terms

An op whose time is set by DRAM traffic does not speed up when AICLK rises, so it lands in the
clock-immune term by construction; an arithmetic-bound op lands in the clock-scaled one. The floor
carries exactly that split, and the two partition it (4.2236 + 10.8077 = 15.031, asserted).

| | modelled | measured | |
|---|---|---|---|
| clock-immune — traffic-bound ops vs `F` | 4.2236 s | 3.9830 s | 94.3 % |
| clock-scaled — arithmetic-bound ops vs `W/f` | 10.8077 s | 10.8630 s | 100.5 % |

**This is not a validation and must not be quoted as one.** Every input to the floor carries a
different problem:

- the per-shape achievable rates were measured **on pc**, whose card runs custom 130-core firmware,
  not on qb2's 110-core p300c;
- `cell_scale = 0.7011` scales a 24.644 s profiled fold onto the 17.34 s cell, which is itself the
  unrecorded ~1063 MHz number;
- `stream_GBps = 424.7` and `dense_cube_TFLOPs = 104.93` carry no recorded clock, and achievable
  bandwidth is not automatically clock-clean either, since the dataflow RISCs that issue the NoC
  transactions run at AICLK.

Where a measurement and a model agree this closely on inputs this dirty, the honest reading is that
it may be structural or it may be two errors cancelling. Saying which is `c10-fold-census`'s job.

## It cuts against `size_scaling/`, and that is worth saying plainly

[`../size_scaling/`](../size_scaling/) reads the two measured sizes as evidence of a large
size-independent work term. If instead `W` is set by per-shape arithmetic roofs, `W` should scale
the way the FLOPs do — and it does not. The two reconcile only if the achievable rate **falls** at
298 aa:

| FLOP model | 298 aa would have to achieve |
|---|---|
| N^1 | 82 % of 512 aa's rate |
| N^2 | 48 % |
| N^3 | 28 % |

A 512×512 pair tensor is 16×16 = 256 tiles over a 110-core grid, about 2.3 tiles per core. At
298 aa it pads to 320 and becomes 10×10 = 100 tiles — under one tile per core. Severe under-fill at
298 aa is not speculative, it is what those tile counts mean, and 48 % is a plausible amount of it.

So this is **evidence against the size-independent term being real work to delete**, and it raises
the prior that `c10-size-scaling` comes back "artifact". It does not settle it, because the
arithmetic-roof picture it rests on comes from shape rates measured on the wrong machine. The row
is running; its per-leg exponents at 640 and 768 aa decide it.

## Handed to `c10-fold-census`

- Re-measure both roofs on qb2 at a recorded 1350 MHz. The arithmetic half cannot be carried from
  pc's 130-core firmware to a 110-core p300c.
- Report the traffic-bound / arithmetic-bound split of measured device time, because that split is
  what `F` and `W` are.
- Do not quote `floor_s = 15.031 s` or `prize_s = 2.309 s` again without re-deriving them.

<!-- -->

    python3 floor_vs_measured.py                      # writes floor_vs_measured.json
    python3 -m pytest test_floor_vs_measured.py -q    # 13 controls
