# B17 raw folds — `TT_BIO_TRIMUL_OUT_L1` at the fold, qb1 card 1, 2026-09-14

Provenance: these seconds were read out of `b1_fold_ab_qb1.json` while the run was live on qb1.
qb1 lost mains/standby power at 22:34:01Z mid-run (host, BMC and all three tailscale endpoints
dropped in the same second), so that JSON and the worktree it sat in are gone. This file is the
transcription, not harness output. Re-running is ~25 min per the invocation in
`b1_fold_ab_512.py`'s header; do that rather than trusting this file if the number matters.

Setup: Blackhole p150a, grid 11x10, ttnn 0.67.4, one device open, one model load, seed 0,
200 sampling steps, 3 recycles, arms interleaved A,B,A2 per rep, 3 reps, one discarded cold fold
per arm. `ps` clean and `lsof /dev/tenstorrent/*` empty before start, 4 idle cards, loadavg 0.18
before the run. Path selector at every size: `trimul_out_l1=False` (arm A), `l1_max_seq=352`,
chunk 32, result memory config DRAM — all four sizes are on the DRAM side, which is the side
this flag is about.

## fold_s, in run order

| size | A | B | A2 | median A / B / A2 | B/A | A/A floor |
|---|---|---|---|---|---|---|
| 384 aa | 11.977 / 12.012 / 11.943 | 11.879 / 12.117 / 12.139 | 11.906 / 11.726 / 11.765 | 11.977 / 12.117 / 11.765 | **0.9884x** | -1.770 % |
| 512 aa | 16.916 / 16.892 / 17.010 | 17.124 / 16.759 / 17.056 | 17.009 / 17.162 / 17.116 | 16.916 / 17.056 / 17.116 | **0.9918x** | +1.182 % |
| 640 aa | 24.581 / 24.670 / 24.938 | 24.775 / 24.711 / 24.796 | 24.592 / 24.946 (3rd never ran) | 24.670 / 24.775 / — | **0.9958x** | +0.40 % |
| 768 aa | host died on this leg | | | | not measured | |

## CIF digests

One hash per size, identical between arm A and arm B: 384 aa `51c99a8e9075106d` (12/12 folds),
512 aa `2f2f5faae481337e` (12/12), 640 aa `b06ebf68a8869340` (11/11). The flag is byte-identical
ON vs OFF, as a memory-config change should be, and 35 folds with zero digest spread also clear
this card of Protenix-v2-class nondeterminism.

## The bias that had to be fixed before any of this counted

The harness warmed arm A only. Arm B moves a memory config, so it compiles trimul programs arm A
never asks for, and that JIT landed inside B's first *timed* fold. At 384 aa / 4 steps it read
A 7.336 s, B 9.425 s, A2 7.361 s — a 0.778x "regression" that was entirely compile. With a cold
fold per arm the same flip reads 0.988x. Any arm added to this harness needs its own warmup.
