# Kernel scout: OpenDDE StructuralTokenExpander

## Result

`StructuralTokenExpander` is the one novel compute block OpenDDE adds on top of the
Protenix-v2 trunk/diffusion/confidence stack (already covered by the closed scouts). It runs
**once per fold**, on the residue-to-structural-token boundary, before diffusion takes over on
the structural-token axis. Profiling it standalone and inside a real end-to-end fold on real
`opendde.pt` weights lands it at **2.21% of a production 7ROA fold** (0.255 s of 11.56 s,
10 cycles / 200 steps / 1 sample). The block is host/upload-bound, not device-compute-bound:
the actual role-pair projection matmul is 0.6% of the block, while host-to-device upload is
68.7% and the host-side pair-init-bias embedding gather is 21.2%. There is no device-kernel
fusion lever, and the Amdahl ceiling even if the whole block were free is 1.023x. **No change.**

This closes the last unprofiled surface the OpenDDE port adds. Everything else in the port
reuses the Protenix-v2 stack verbatim (already closed: TriangleMultiplication, pair-transition
SwiGLU, FC2+residual, TriangleAttention, OuterProductMean).

## Hardware and method

pc physical card 0, one Blackhole P150, bf16, HiFi4 / fp32 dest-acc. Real `opendde.pt` weights
for the e2e share and the real-weight standalone breakdown; the random-weight standalone sweep
uses identical shapes/dtypes (device op timing is shape-dependent, not value-dependent, and the
real-weight standalone at Ns=229 cross-checks the embedded number: 0.259 s vs 0.255 s). Every
timed run is warm (program cache primed by a discarded reduced fold) and ends with a device
synchronize. Reproduce:

```bash
PYTHONPATH=. TT_VISIBLE_DEVICES=0 /home/moritz/tt-bio/env/bin/python3 \
    scripts/opendde_structtoken_scout_bench.py --sizes 229 275 512 1024 --chunk-sweep
PYTHONPATH=. TT_VISIBLE_DEVICES=0 /home/moritz/tt-bio/env/bin/python3 \
    scripts/opendde_structtoken_scout_bench.py --real-weights
OPENDDE_NCYCLES=10 OPENDDE_NSTEP=200 PYTHONPATH=. TT_VISIBLE_DEVICES=0 \
    /home/moritz/tt-bio/env/bin/python3 scripts/opendde_structtoken_share.py
```

Realistic structural-token counts were measured from real targets via
`build_structural_token_features` (2026-07-12): 7ROA 117 res to Ns=229, hemoglobin 141 res to
Ns=275. The protein-only scope is the only case the port supports today (`opendde_data.py`).

## Real share of a production fold

`scripts/opendde_structtoken_share.py` instruments `OpenDDE.fold` to time the expander, the
expander+refiner seam, the residue-axis trunk, and the EDM sampler separately, on 7ROA with real
weights. The expander is called once; the trunk runs per recycle cycle; diffusion runs per step.

| setting (cycles / steps / 1 sample) | total fold | expander | expander share | trunk | diffusion |
|---|---:|---:|---:|---:|---:|
| 2 / 20 (reduced) | 2.469 s | 0.255 s | 10.3% | 1.461 s (59.2%) | 0.414 s (16.8%) |
| 10 / 200 (production) | 11.559 s | 0.255 s | **2.21%** | 7.202 s (62.3%) | 3.769 s (32.6%) |

The expander+refiner seam (the expander plus the 4-block structural-token Pairformer refiner) is
0.414 s, 3.58% at production. The expander alone is the 0.255 s / 2.21% figure. Going from
reduced to production settings grows the trunk ~5x and diffusion ~9x while the expander stays
flat at one call, so its share falls from 10.3% to 2.21%. Production is the honest number.

Amdahl ceiling: even if the entire expander became free, max fold speedup is
1/(1-0.0221) = **1.023x**.

## Per-op breakdown (real weights, 7ROA, Ns=229, pair_chunk_size=128)

`scripts/opendde_structtoken_scout_bench.py --real-weights` wraps every device op and phase
method with synchronized timing. Categories are nested (phase totals include the uploads that
happen inside them), so the phase columns are the meaningful split; the device-op column is the
sub-component view.

| phase | time | share of block | notes |
|---|---:|---:|---|
| host-to-device upload (`_up`) | 0.1781 s | 68.7% | 26 uploads: activations, per-chunk pair tensors, 5 attn-bias terms/chunk |
| pair projection + scatter | 0.0612 s | 23.6% | the 49-role-pair path (see below); includes its internal upload |
| pair-init-bias (host gather) | 0.0550 s | 21.2% | sum of 5 host embedding index-selects, pure CPU |
| attn-bias assembly | 0.0026 s | 1.0% | scalar-weighted mask sum on device |
| role-pair matmul (`ttnn.linear`) | 0.0015 s | **0.6%** | the actual device matmul compute |
| split-MLP layernorm | 0.0001 s | <0.1% | |
| **total** | **0.2591 s** | | |

The device matmul compute is 0.6% of the block. The block is dominated by moving tensors
host-to-device and by a host-side embedding gather, not by anything a device kernel could fuse.

## The three scout questions

### (a) Are the 49 role-pair projections 49 separate dispatches?

No, not in the protein-only scope. `_pair_project_full` groups the flattened pair positions by
`role_i*7+role_j` and runs one matmul per non-empty group. For a protein-only input only the
protein role-pairs are present (protein_bb/protein_sc), so ~5 groups per chunk are non-empty,
not 49. Measured `ttnn.linear` call count: 10 at Ns=229 (2 chunks x 5), 18 at Ns=427 (4 chunks),
30 at Ns=853 (7 chunks). The 49-projection dispatch concern is moot for the current scope; it
would only materialize for a mixed protein+DNA+RNA complex, which the port does not yet support.

Even at 49 groups the device matmul time would still be negligible: it is 0.0015 s today for
5 groups, i.e. ~0.3 ms/group, so 49 groups is ~15 ms, still under 6% of the block and under
0.13% of a production fold. Batching the groups into one matmul would save sub-millisecond.

### (b) Does pair_chunk_size=128 cause avoidable DRAM round-trips?

No. A chunk sweep (pair_chunk_size=128 vs a single chunk = Ns) at four sizes shows single-chunk
is within noise of chunked: speedup 0.977x to 1.033x (chunked over single, so >1 means chunking
is slower). At Ns=229 real weights: 0.2591 s chunked vs 0.2552 s single, a 1.5% difference. The
chunking is over rows and the per-chunk work is independent; it neither exploits L1 headroom nor
pays a measurable DRAM penalty. No lever from changing the chunk size.

| Ns | chunked (128) | single chunk | chunked/single |
|---|---:|---:|---:|
| 190 | 0.167 s | 0.171 s | 0.977 |
| 229 (real weights) | 0.259 s | 0.255 s | 1.015 |
| 427 | 0.824 s | 0.814 s | 1.012 |
| 853 | 3.245 s | 3.170 s | 1.024 |

### (c) Can the bias-add epilogue fuse into the preceding matmul?

No. The pair-update is `z = z + pair_project(z) + pair_init_bias`, where `pair_init_bias` is a
per-(row, col, channel) tensor assembled on the host from five embedding index-selects
(`same_parent`, `same_residue_twin`, `prev_bb`, `next_bb`, `role_pair_type`), shape
`(clen, Ns, c_z)`. A matmul `bias=` epilogue takes a 1D vector over the channel dim, not a
per-position tensor, so this bias cannot fuse into the projection matmul. The preceding matmul
is also only 0.6% of the block, so fusing its epilogue would save sub-millisecond regardless.

The pair-init-bias cost is 21.2% of the block, but it is host compute plus an upload, not a
device add. Removing it would mean moving the embedding gathers onto the device (the index
tensors are host integers that would need uploading anyway) for a block that runs once per fold.
That is a host-upload restructure, not a device-kernel fusion, and its absolute payoff is
~0.05 s per fold (0.4% of a production fold). It does not clear the bar for an
accuracy-sensitive kernel path.

## Resolution

StructuralTokenExpander is at the dispatch/Amdahl ceiling. It is 2.21% of a production fold,
host/upload-bound (68.7% upload, 21.2% host gather), with the device matmul compute at 0.6% of
the block. The 49-projection concern collapses to ~5 for protein-only, chunking is neutral, and
the bias-add cannot fuse into the matmul. The maximum possible win from making the entire block
free is 1.023x wall-clock, and no device-kernel fusion captures any meaningful fraction of that.
No fusion is prototyped; no code changes. This is a documented ceiling, not a missed lever.
