# The token DiT's 80.53 GB/fold was collected on 2026-09-11. This row is a pass.

Kill criterion 1 of the brief: "Kill if the round trip is already collected at the tip: report the
byte accounting and STOP, which is a pass." It is collected. Below is the accounting, and the
measurement that says so on a real fold rather than from reading the source.

## The answer

| | |
|---|---|
| of the 80.53 GB/fold still on the table at the tip | **0.00 GB/fold** |
| op ratio available on `DiffusionTransformerLayer\|1x512x768` | **1.000x** — there is no arm to switch on |
| fold ratio available | **1.000x** |

`BOLTZ2_TOKEN_DIT_SDPA` defaults **on** (`tt_bio/tenstorrent.py:1336`). `fc7fed56f` flipped it
2026-09-11 19:43 UTC, four and a half hours after `graph_512_all.json.gz` — the capture the 80.53
GB/fold row was ranked from — was taken at 15:04 UTC. The row ranks a configuration `main` has not
run since that evening.

The brief's second candidate, the atom transformer at `1x140x32x128`, was never gated at all:
`AttentionPairBias._attention` (`:7386`) calls `ttnn.transformer.scaled_dot_product_attention`
unconditionally for any non-fp32 dtype. It has no flag and no eligibility condition to relax.

## Measurement 1: the lever serves every call, at every size

`scripts/lever_census.py` through the real CLI and its worker spawn, summed across processes
(a counter read in the launcher is always zero — the fold happens in a spawned worker).
Boltz-2, `--single_sequence`, 6 sampling steps, seed 0, whglx card 3 (Wormhole).
`perf/roof_dit_sdpa/out/census_boltz2_*_whglx3.json`.

| size (aa) | resolved | served | declined |
|---|---|---|---|
| 128 | True | 144 | 0 |
| 256 | True | 144 | 0 |
| 298 | True | 144 | 0 |
| 512 | True | 144 | 0 |
| 640 | True | 144 | 0 |
| 768 | True | 144 | 0 |
| 1024 | True | 144 | 0 |

144 = 24 token-DiT layers x 6 steps. At the production 200 steps that is 4800, which is the call
count of record, and **none of them decline at any size on the ladder**. That closes kill criterion
4 the other way: this is not a one-size lever. Note the contrast in the same census runs —
`TRIATT_PERSISTENT_MASK` serves 560/560 at 256-512 aa, declines 560/560 at 128 aa on `memory`, and
splits 560/560 at 640 and 768 aa on `pm_over_l1`. That is what a size-conditioned gate looks like in
this table, and the token DiT SDPA is not one.

Every committed census in `perf/sizegate/` was taken while the flag was still off
(`resolved: False`, `off-by-design`), so none of them could have shown this.

## Measurement 2: the two shapes issue one fused op each

Per-site SDPA census (`SDPA_GRID_Q_CHUNK_CENSUS`), 512 aa, 6 steps, same card.
`perf/roof_dit_sdpa/out/sdpa_sites_512_whglx3.json`.

    site        q_len  k_len   d   work   calls
    token_dit     512    512  64     16     144
    atom           32    128  32    560      36

    ragged tails: tri_att [0 ragged, 560 clean], attn_pair_bias [0, 36], token_dit [0, 144]

`token_dit` at q=k=512, d=64, work=16 is the 16-head attention over 512 tokens. 512 x 512 x 16 in
bf16 is 8.389 MB — the exact matrix the published row said was written to DRAM and read straight
back. It is one `scaled_dot_product_attention` call, so the matrix lives in a circular buffer and is
never allocated. `atom` at 36 calls (6 per step, 1200 at 200 steps) is the second candidate, also
one fused op. Both shapes are already served.

## The conversion the brief asked for, and why the row ranked higher than it was worth

80.53 GB/fold against the measured 424.7 GB/s stream roof is **0.190 s**. The budget table's deficit
on that row is 1.448 s, so collecting every byte would have closed **13 %** of it, or 1.09 % of the
17.340 s cell.

The flag was worth more than that when it was flipped: 22.195 -> 21.095 s, 1.0522x
(`perf/b2x-integrate/README.md`, N=3 per arm, A/A floor 0.19 %). 1.100 s against a 0.190 s byte
prize means **most of that win was not bytes** — it is 4800 fewer program pairs and a softmax that
never leaves L1. The byte ranking pointed at the right site for the wrong reason. Bytes rank; they
do not price.

## What is left at this site

The 8.389 MB rollout-invariant pair bias, re-read as the SDPA mask on all 4800 calls: **40.3
GB/fold, 0.095 s at the stream roof**. It is not a round trip — the mask is a genuine operand, read
once per call by a kernel that has to read it. Halving it in bfp8 is already measured and refused on
accuracy (0.949x AND 1.496 A), and the campaign asked not to rediscover that. Anything else needs
the bias resident across calls, which is 8.389 MB of L1 held for the whole rollout, and that is a
different lever from this one.

## Architecture note

Nothing here is an eligibility lever, so the usual Wormhole-to-Blackhole non-transfer warning does
not apply to the verdict: served/declined is control flow, not an L1 fit, and it reads the same on
either architecture. Where an architecture would matter is the 0.190 s conversion, which uses the
Blackhole roofs the budget table was measured on. The census above ran on Wormhole because that is
this row's card grant; it ranks, and there is nothing left to price.

## Parity

No model code changed on this branch. Nothing to gate.

## The brief's third question: does `sdpa_generic.plan_for_shape` refuse this shape?

Moot at this site. `sdpa_generic` has exactly one caller, `tt_bio/triatt_sdpa.py`, and that is the
triangle attention. The token DiT and the atom transformer both call the stock
`ttnn.transformer.scaled_dot_product_attention` and never reach it. Routing them through it would
change nothing either way: `sdpa_generic` is a transcription of the same wheel kernel, and the
sibling capture measured it accepting the token DiT's shape anyway (19 of 20 chunk pairs, no padded
mask, 128 of 130 cores, 114 KB of CB). There is no refusal to lift.
