# `wk/land-standing` is ready to land, and one item on it is a correctness fix others can quote

As of 2026-09-26. **0 behind `origin/main`**, so merging is a fast-forward and `main` afterwards
is exactly the tree the suite below scored.

    tip    74c13e5c3
    tree   6e33abef952d63f121ee248bed665bcb3425e163
    suite  4722 passed, 282 skipped, 0 failed in 588 s   (tree hash read after the run)

45 files, +5427 / -43. **The only shared-surface file is `tt_bio/tenstorrent.py`, +10 lines, and
it is comment-only** — verified by filtering the diff for non-comment additions and getting
nothing.

## The one item that should not wait

**`perf/bcx_tapedfwd/khole.py` had two call errors and produced a plausible-looking accuracy
column that was wrong on both board classes, for as long as the harness has existed.** It passed
`scale ** -1` into a parameter that is a softmax MULTIPLIER, and called the fall-back with
`bias_scale_inv=1.0` while the fused path reads the bias as pre-baked. So its two device arms
were not computing the same function as each other (`fused_vs_fallback` 56.6 %) and neither
computed the reference's. Any row reading that column would have taken a bogus grade.

Fixed, and the corrected instrument now reports the fused route as **8.7-12.6 % closer to float64
than the path it replaces** at six lengths. A row carries `reference_usable` /
`reference_rms_ratio` and prints REFERENCE NOT USABLE when the shipped fall-back sits outside
[0.5, 2] of the reference, so the same class of error announces itself next time.

## Everything else, by what it is for

| file | what it settles |
|---|---|
| `DIVIDING-K-VERDICT.md` | `TT_BIO_TRIATT_DIVIDING_K`: reach is ONE length (832, a hole between 704 and 1088 which both serve), kernel accuracy favourable, structural accuracy not answerable on any fixture this box has, speed ~1.5x and unpriced. Recommendation: keep it off |
| `BACKLOG-AUDIT.md` | every candidate the charter names, checked with `merge-base` and the real default in source: all landed, contained or closed |
| `EXTRAMSA-BACKWARD-DEFECT.md` | `predictor(extra_msa=True)` raises `Buffer is not allocated`; root cause, a seconds-long repro, and an audit showing the blast radius is one of five checkpoint call sites |
| `checkpoint_capture_repro.py` | model-free red/green for that defect — seconds, against ~10 min for a BC2 round |
| `offsweep.py` | enumerates every default-OFF `env_flag` and asks whether the source says why. 31 flags, no landable inference win |
| `gate_reach.py` | crosses the gate's declared rungs with a lever's changed lengths and the per-site defaults |
| `obh_grid.py` | prices `out_block_h = 5` across four core grids; it transfers |
| `run_arm.py` | `--binder-length`, `--no-levers`, `--shipped` / `--multimer-pool`, `--pool-resident` |
| `hifi_dump_sitecustomize.py`, `mkapo.py`, `khole_summary.py`, `domain_rmsd.py`, `rmsd_matrix.py` | the instruments the above were taken with |

## What is NOT on it

No default flips and no perf claims. Nothing here moves a shipped number, which is why the row's
`TOTAL` is unchanged at 22.6335 s.
