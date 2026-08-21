# Does the protenix template embedder's L1-resident pair tensor starve its consumer?

`Protenix._ln(..., l1=True)` (tt_bio/protenix.py) is the third call site of the pattern that
crashed Boltz-2 at 704 padded tokens: `_l1_layer_norm(x, 1.5)` prices a pair tensor's L1
residency as a multiple of the tensor against the whole grid's aggregate L1, while the wall its
consumers hit is per core. The site is live on every protenix-v2 fold -- `dummy_template_features`
emits 4 template slots unconditionally (`tt_bio/protenix_data.py:458`), so `_template` always
runs.

Answer: **it does not, at any shape the gate admits, on any grid tt-bio ships on.** Two
independent measured reasons, both of which were the opposite at the Boltz-2 site.

All figures are B/bank, p150a, `_l1_bank_bytes()` = 1461760 idle,
`get_max_worker_l1_unreserved_size()` = 1532416, 130 banks.

## 1. The window opens with L1 empty

`fold_l1_trace.py` wraps `_l1_layer_norm` and `_narrow_proj_linear` in the real trunk and reads
the allocator at the residency window.

    TT_VISIBLE_DEVICES=0 python3 perf/protenix_tpl_l1/fold_l1_trace.py --n 496 --cycles 10 --steps 8

| tokens | cycles | free at window open | after the norm | after the projection | out | fold |
|---:|---:|---:|---:|---:|---|---|
| 120 | 2 | 1461760 (idle) | 1400320 | 1383936 | L1 | ok |
| 496 | 10 | 1461760 (idle) | 460288 | 208384 | L1 | ok, 108 s, plddt 0.351 |
| 505 | 10 | 1461760 (idle) | 441856 | 185856 | L1 | ok, 89 s, plddt 0.356 |
| 506 | 10 | 1461760 (idle) | 439808 | 183808 | L1 | ok, 108 s, plddt 0.351 |
| 512 | 10 | 1461760 (idle) | not admitted | -- | DRAM | ok, 66 s, plddt 0.359 |

506 is the boundary, and the real fold reads 439808 / 183808 -- byte-for-byte what the isolated
probe below predicts with no ballast. That is what makes the ballast sweep a measurement of the
real fold rather than of an idle device. 512 is the first rung the gate refuses; it folds too.

Free at window open is the full idle bank at **every** cycle. Boltz-2's crash needed 247 KB/core
of other live L1 buffers on top of the pair tensor; here there are none, because `_template`
takes its projections above the block loop and frees the tensor before the first
PairformerLayer.

## 2. At the worst shape the gate admits, the consumers fit with room left

The gate admits L1 for token counts up to **506** at c_z=256 (and 415 at OpenDDE's c_z=384),
monotone in token count, so the boundary is where the consumers have least room. `probe.py`
replays the window in isolation -- which measurement 1 shows *is* the real state -- and starves
it with a synthetic L1 ballast standing in for other live buffers.

    TT_VISIBLE_DEVICES=0 PROBE_BALLAST=0,131072,163840,196608,212992,245760,278528 \
        python3 perf/protenix_tpl_l1/probe.py 506

| ballast | norm -> L1 | free after norm | projection output | outcome |
|---:|---|---:|---|---|
| 0 | yes | 439808 | L1 (256000), CBs fit in 183808 | ok |
| 131072 | yes | 308736 | L1, CBs fit in 52736 | ok |
| 163840 | yes | 275968 | L1 refused, caught -> DRAM | ok |
| 196608 | yes | 243200 | DRAM | ok |
| 212992 | yes | 226816 | DRAM | ok |
| 245760 | yes | 194048 | DRAM | ok |
| 278528 .. 438272 | **no** (norm's own L1 refused, caught) | -- | DRAM | ok |

So the consumer need at the boundary is 256000 for the projection's own L1 output plus static
circular buffers that fit in 52736 -- 308736 total, against 439808 available. And there is no
ballast at which anything throws: the window degrades L1 output -> DRAM output -> DRAM norm, each
step a caught exception, and the norm stops taking L1 while 194048 are still free, 3.7x what the
DRAM-output arm's circular buffers need.

## 3. Grid and pair width

A headroom of h admits at most per_core/h per bank, so it leaves per_core * (1 - 1/h) = 510805 B
free whatever the grid, token count or pair width -- on a real part the banks are the worker
cores. The consumer's peak need is `per_core_M * n_tiles` tiles of output plus grid-independent
circular buffers, so it does not grow with the grid either. Measured at each grid's own worst
admitted shape, with `TT_BIO_FORCE_GRID` for the program-config path and a ballast that brings
free L1 down to what a real part of that core count would have:

| grid | c_z | worst admitted | free after norm | after projection | outcome |
|---|---:|---:|---:|---:|---|
| 13x10 | 256 | 506 | 439808 | 183808 | ok |
| 11x10 | 256 | 457 | 439808 (ballast 157696) | 222720 | ok |
| 8x9 | 256 | 374 | 437760 (ballast 456704) | 294400 | ok |
| 13x10 | 384 | 415 | 441856 | 271872 | ok |
| 13x10 | 384 | 416 | not admitted (DRAM), as predicted | -- | ok |

## Why no reserve was applied

`_PAIR_L1_CONSUMER_RESERVE` is 640 KiB, measured for a softmax needing 563658 B/core. Passing it
here moves the boundary from 506 tokens to 463 and takes the L1 lever (2180.7 -> 853.1 us on the
four-template region) off every fold in between, to fix a clash this site does not have. Only a
reserve inside a 2016-byte window (510055..512070) reproduces today's decision exactly, which is
another way of saying the 1.5 multiplier already *is* a 511 KB/core reserve at this shape -- 1.65x
the measured need. `tests/test_template_l1_consumer_margin.py` pins that, and pins the other
end: the margin is a floor on the headroom, not a ceiling. Below **1.2524** the guaranteed free
drops under the 308736 need. `TRANSPOSE_L1_HEADROOM` was cut 2.5 -> 1.25 on this same helper for
perf, so that is not a hypothetical direction of travel -- and that site carries both an
exception-guard fallback and an opt-in per-core reserve because of it.
