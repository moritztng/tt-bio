# Pre-registered, before the first measurement of this row

Written and committed before anything ran. `b2z2-trunk-byte-floor` left 282.2 MB of the block's
416.3 MB redundancy surface unbuilt and measured that its own 134.2 MB deletion realised **1.76x**
what the DRAM-byte ledger priced it at. This row builds the other two sites and asks whether that
1.76 is a rule or an accident of one multicast `in0`.

## The two sites

| site | removable | what it is |
|---|---|---|
| **R2** trimul `g_out` | 134.2 MB | the fused four-way in-projection and the output gate both read the whole 67.1 MB `x_norm_in`, x2 trimuls |
| **R1b** tri-attention bias | 134.2 MB | the 128->32 pair-bias projection reads the same `x_norm` the fused qkv+gate pass already reads, x2 attentions |

## The DRAM-byte prediction, which is the thing being tested

The byte model says 45.2 % of the block is byte-proportional. Each site is **134.2 MB of 7,417 MB
= 1.809 %** of this trace's own DRAM traffic, so each is worth **1.809 % x 0.452 = 0.818 %** of the
block: **0.702 ms of the parent's 85.884 ms base**, a block ratio of **1.00824x**. That is the
denominator every "realization" below is measured against.

## P1 - R2, the trimul gate

**Predicted block ratio 1.0146x, band 1.010-1.017x. Predicted realization 1.5-2.0, point 1.76.**

Reason: `x_norm_in` is the `in0` of `minimal_matmul`, multicast down a grid axis exactly as the
tri-attention activation the parent deleted was. Same operand class, same width class, so if the
parent's explanation is right this site has to pay the same coefficient. If it does not, the
explanation is wrong and the 1.76 was a property of that one call, not of multicast `in0`s.

## P2 - R1b, the bias projection

**Predicted block ratio 1.008x, band 1.004-1.012x. Predicted realization 0.5-1.4, point 1.0.**

Reason: the bias projection's output is one tile wide. Its `in0` is the same activation, but a
matmul that produces 1 N tile spreads over far fewer N-axis cores than one producing 12 or 16, so
each delivered byte is multicast to fewer receivers and the DRAM-byte ledger is closer to right
about it. **This is the site that discriminates the two explanations**, and it is predicted to
realise materially less than R2.

## P3 - bit-exactness is not a hope, it is the design

Deleting a redundant read cannot change a number. **`torch.equal` True and max abs exactly 0.0** on
every output of both sites, with a negative control that breaks the check. Any deviation at all
means the build is wrong, not that the precision moved.

## P4 - the fused-kernel eligibility census must be flat

`qkv_heads`, `qkvg_heads`, `trimul_tail`, `_pair_proj_minimal_matmul`, the head-major tail and
`reblock_gated`, counted per arm. The earlier byte row's sign flipped because a change made the
tuned kernels decline; a flat census is the only thing that rules that out. **Predicted: identical
counts in both arms at both sites.**

## The falsifier

**If either site's measured block ratio realises below 1.0x of its 1.00824x DRAM-byte prediction,
the under-pricing result does not generalise** and the parent's 1.76 was specific to that one
multicast `in0`. That is a publishable outcome either way: it tells the campaign which operand
classes its currency is wrong about.

Secondary, and a weaker claim: if both sites land at their point predictions the block carries
**~1.023x** from the two together, and the trunk's removable-byte ledger is fully built out.

## Rules this row runs under

No ratio off a watcher-armed run. WH card 10 on whglx, pinned and leased, trace region capped at
512 MiB. Paired interleaved one-process A/B with an A/A floor measured in the same run, per
CONTEXT 4-I, because whglx is loaded and only ratios are admissible from it. Flags default OFF.
