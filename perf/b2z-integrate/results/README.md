# b2z-integrate results — Blackhole, qb2 card 3

Paired, interleaved A/B runs under benchlock. `base` is the shipped incumbent, folded immediately
before every arm, so each rep yields a ratio rather than two absolute numbers off a shared box.

| file | what it is |
|---|---|
| `steps50.json` | **A BROKEN RUN, kept deliberately.** The `cfg:sampling_steps=50` lever never reached the live model, so both arms folded at 200 steps. It reports `paired_speedup 1.00015` and `bit_exact_vs_base true` across 12 folds — a no-op wearing the costume of a measured negative result. Kept as the example of what a false negative looks like in this harness. |
| `steps50b.json` | the run after the fix, refused by the new observed-step guard: base asked for 200 and the diffusion loop reported 191 callbacks, so the absolute tolerance was wrong. Recalibrated to a relative 0.75-1.25 band. |
| `steps50c.json` | the real measurement. base 19.693 s vs steps50 16.107 s, 5 reps each, **1.2226x**, ratios 1.2215-1.2336, `fold_spread_pct` 1.06 %. Observed steps 191 vs 41. |
| `integ.json` | both landed levers together (`TT_BIO_FUSE_BIAS_STACKS`, `TT_BIO_MSA_DEPTH_LADDER`) plus the 50-step protocol, `--combined all` for additivity. |

The `.nohup` files are the full stdout of each run, including the benchlock wait and the refusal
above.

**Why the broken run is in the repo.** The failure it records is the one this harness exists to
prevent: an instrument that returns a plausible number while measuring nothing. Deleting it would
leave only the runs that worked.
