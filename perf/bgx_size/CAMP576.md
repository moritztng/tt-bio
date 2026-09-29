# The 576-token campaign, end to end

The only BindCraft 2 campaign that has run to completion above 288 tokens. `docs/bindcraft2.md`
quotes it in "What fits"; this is where those numbers come from. Run by `camp576_watch.sh`, which
recorded the outcome after the launching turn had already ended.

`perf/bgx_size/out/` is gitignored, so the raw tree lives on qb1 at
`~/.coworker/wt/bgx-size/perf/bgx_size/out/camp544/` and only the figures below travel.

## What ran

| | |
|---|---|
| target | hIL2R, 387 residues, 2 chains |
| binder | 146 residues |
| token axis | **576**, the Evoformer seam's own axis. Filed under `camp544`, which is the target-plus-binder label, one bucket low |
| card | qb1 p150a, card 1, `0000:41:00.0` |
| commit | `509fea2d5`, clean, on `origin/main` |
| seed | 100, `validation="device"`, `--trajectories 1` explicit, cap 2 |

The count was set explicitly rather than left to `auto`: at the time this launched, the auto
default was the guard `bgx-traj` was investigating. `rung.json` records what auto *would* have
picked as **3**, from `11d3fd8eb` — three trajectories priced at 29.9 GB each against a card with
29.1 GB for them. That is the defect `bgx-traj` fixed, caught at the top of the supported range.

## Outcome

Stopped on its trajectory cap, not on an error. `error: None`, `finished_utc`
2026-09-29T16:18:00Z, **13,121.85 s** of wall clock, 3 h 39 m.

```
campaign done: 0 accepted design(s) after 2 trajectories, ranked by i_pDAE
```

Trajectory 1 passed screen, refine and mutate (i_pTM 0.87 / 0.86 / 0.76), drew 10 MPNN redesigns
and **0 of 10 passed**. Trajectory 2 ran and produced no row in `!_Trajectories.csv`. There is no
`3_Ranked/!_Ranked.csv`: nothing was accepted. Two trajectories put a rate on nothing, and the
docs claim none above 288.

## Timing and clock

249 round-to-round deltas over 250 sampled rounds: **median 47.20 s**, p10 46.95, p90 47.46. That
is the same round the ladder measured at this axis on a quiet card, 47.38 s.

AICLK sampled during the run, `n=12,461`: **median 1350 MHz**, max 1350, min 800. Box load1 2.5.

## Memory

Host high-water 19.07 GB (`VmHWM` 20,477,931,520 B) with 503 GB available, so the box was never
close. Device free at round boundaries ran 31.87 GB down to 29.20 GB; that sampler fires between
rounds and does not see the in-seam peak, which is the ladder's 14.23 GB.

## Where the fused kernels went

Both counters for the whole campaign:

| kernel | served | declined |
|---|---|---|
| `triatt_fused_hifi` (forward) | 81,000 | 15,232 |
| `triatt_bw` (backward) | **0** | 27,000 |

The forward serves five calls in six. The backward serves none, and it is not a routing problem:
`bw_calls` is 27,000, incremented unconditionally before any gate (`tt_bio/autograd.py`), so the
kernel was reached every time and its shape gate refused every time. Per-round it is 108 calls
from round 2 on, and there is no round in the 249 where the count failed to advance.

This matches the ladder's per-axis census exactly: the fused backward serves 108 of 108 at 288 and
0 of 108 at 352, 416, 448, 480, 512 and 576. `tt_bio/triatt_bw.py` names `q_chunk_tiles` as the
knob that would pull the whole-query form back under the L1 line at a longer axis, and nothing
sets it. Serving the backward above 288 is unclaimed perf work, not a correctness problem: the
chunked recompute is correct at every shape.
