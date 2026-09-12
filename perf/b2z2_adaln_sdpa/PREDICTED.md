# b2z2-step-adaln-sdpa — pre-registered, before this row's first device run

Written against the committed evidence only: `perf/b2z2_next_sites/site_rank_wh_c2.json` (the
ownership-annotated ranking this row is handed), `state/b2z2-step-layernorm-fusion.md` §1 (the
114-LayerNorm per-call join), `state/b2z2-step-binaryng-fusion.md` (the elementwise NO-GO), and
`tt_bio/tenstorrent.py` at `9cd6c2ff1`. No number below has been measured by this row yet.

Base for every ratio: the settled 512 aa `Diffusion.__call__` on Wormhole, which three rows have
now reproduced at 41.50-41.60 ms/step. The cell's own step A/A floor is ~1.001x.

## P0 — the brief's 2.902 ms/step for "AdaLN norm of a" is a cost-cluster, and half of it is owned

`site_rank.py` clusters by op code and cost. Its 97-program `LayerNorm` cluster at 2.9024 ms has to
contain the 48 `AdaLN.s_terms` norms (31.94 us each) as well as the 48 `AdaLN.__call__` norms
(27.88 us each): the two are 4 us apart on the same `[1, 512, 768]` shape, so no cost clustering can
separate them, and `b2z2-step-layernorm-fusion` already **owns the s half** (it is the 1.5332 ms it
deletes). The site's four LayerNorm clusters sum to 3.2957 ms over 114 programs, which is that row's
per-call join exactly, so the join is the authority and the cluster is not.

**Predicted unowned population: 60 programs, 1.566 ms/step** — 48 token-stack `AdaLN.__call__`
norms at 27.88 us (1.3381 ms) plus 12 atom encoder/decoder norms at 18.97 us (0.2276 ms).
**That is 3.8 % of the step, not 7.0 %, and it caps the whole site at 1.039x even if the norm went
free.** Falsifier: my own per-call join on my own build disagrees with 60 / 1.566.

## P1 — the fold the brief proposes cannot be built, and the half that can was already measured slower

The shipped form is not `norm(x) * (1+scale) + shift` with foldable affine. It is

    a = ttnn.layer_norm(a, epsilon=1e-5)                              # NO weight, NO bias
    a = ttnn.multiply_(a, s_scale, input_tensor_b_activations=[SIGMOID])
    a = ttnn.add_(a, s_bias)

and `s_scale` / `s_bias` are per-TOKEN `[1, 512, 768]` projections of `s`, not per-channel vectors.
`ttnn.layer_norm`'s `weight=` / `bias=` slots take a 1-D per-channel term, so there is nothing to
fold: the norm carries no affine precisely because AdaLN's affine is the adaptive one. The
multiply+add half is **already built and already measured** — `b2z2-step-binaryng-fusion` replaced
it with one `ttnn.mac`, bit-exact under `torch.equal`, and the step read **0.99315x**.

**Predicted: the brief's step-2 lever is structurally unavailable and this row will not build it.**
Falsifier: if ttnn 0.68.0's `layer_norm` accepts a full-rank per-row `weight` and returns the same
values, P1 is wrong and the fold is a 48-program deletion worth ~1.032x by program count alone.

## P2 — the norm of `a` is parallelism-starved, not byte-bound, and that is what is left of the site

`[1, 512, 768]` bf16 in and out is 1.572 MB in 27.88 us = **56.4 GB/s**, roughly a quarter of the
byte roof the same block's matmuls reach 23-47 % of. 512 rows is 16 tile-rows, and ttnn's
interleaved LayerNorm hands whole tile-rows to cores, so I predict the op runs on **16-24 of the
72 cores** and is waiting on nothing but its own core count.

A row-parallel form (height-sharded input, sharded LN program config) keeps the reduction inside a
row and so should be **bit-exact**; what it can cost is the shard/unshard round trip.

**Predicted step ratio 1.005x - 1.020x, best point 1.010x**, with the 1.039x whole-site ceiling from
P0 above it. **Falsifier: a measured core count of 48 or more refutes the mechanism outright.**
**Falsifier: a step ratio inside the ~1.001x A/A floor makes this not a lever at this size** — and
then the size it becomes one at is where the norm's row count stops covering the grid, i.e. it is
already as parallel as it will get and the answer is that the site is not a lever at ANY size.

## P3 — the token SDPA's 2.745 ms is bytes, and three quarters of the bytes are the bias

Per call, `[1, 16, 512, 48]` q/k/v against a `[1, 16, 512, 512]` bf16 additive bias:

| term | per call | share |
|---|---|---|
| arithmetic | 805.3 MFLOP -> **7.0 TF/s at 114.39 us, 3.9 % of the measured 178.6 TF/s roof** | not the cost |
| q + k + v + out bytes | 3.15 MB | 27 % |
| **the bias** | **8.39 MB** | **73 %** |
| per-program constant | 9.76 us of 114.39 | 8.5 % |

Total 11.5 MB at 114.39 us = **100.5 GB/s**, and 24 calls a step is **276 MB/step, 55 GB a fold, of
a tensor that is constant across all 200 steps**. **Predicted split: ~4 % arithmetic, ~8.5 % per-
program constant, the remainder byte-bound and dominated by the bias.** Falsifier: the bias under
half the bytes, or a measured wall within 20 % of the arithmetic roof.

This is the ruling wave 1 wrote for the TRUNK's SDPA re-derived for the SAMPLER's, and it comes out
the other way: the trunk's was called packer passes, this one is an operand it re-reads.

## P4 — so the SDPA lever is the bias's dtype, not the algorithm, and it is not bit-exact

If P3 holds, chunking or batching the 24 layers is the wrong lever (the per-program constant is
8.5 % of the site, 0.23 ms/step, inside the fold's noise), and the right one is to stop moving
8.39 MB of bf16 bias 4800 times a fold. bfp8_b is 0.531x the bytes on the term that is 73 % of them:
predicted **1.015x - 1.035x on the step, best point 1.024x** (276 -> 182 MB/step at the same
100 GB/s is 2.745 -> 1.82 ms).

**NOT bit-exact** — it is a storage-precision change on the attention bias — so it scores at 298 aa
and 512 aa per pseudo-domain or it does not ship. Falsifier: under 1.005x refutes the byte mechanism
on this op, and then P3's split is wrong somewhere I have not looked.

## P5 — the two sites are disjoint from every lever this wave is composing

`TT_BIO_ATOM_L1`, `TT_BIO_ATOM_KEY_WINDOW` and `TT_BIO_ATOM_KV_PREPROJ` are all in the ATOM branch;
the layernorm row's 48 norms are `AdaLN.s_terms`. P2 is `AdaLN.__call__` and P4 is the token SDPA,
neither of which any of those four touch. The one real overlap risk is P2 against the layernorm
row: both are `ttnn.layer_norm` calls, different call sites, and stacking them is additive only if
the s-norm hoist does not change the `a` norm's core occupancy. **Predicted: additive to within the
A/A floor.** Falsifier: a measured `both` arm that differs from the product of the two.
